"""The implementation of U-Net and FCRN-A models."""
from typing import Tuple

# external imports
import numpy as np
import collections
import torch
from torch import nn
import torchvision
from torchvision import models
from skimage import morphology as morph
import torch.nn.functional as F

# interal import
import arguments as a
import config as c

def conv_block(channels: Tuple[int, int],
               size: Tuple[int, int],
               stride: Tuple[int, int]=(1, 1),
               N: int=1):
    """
    Create a block with N convolutional layers with ReLU activation function.
    The first layer is IN x OUT, and all others - OUT x OUT.
    Args:
        channels: (IN, OUT) - no. of input and output channels
        size: kernel size (fixed for all convolution in a block)
        stride: stride (fixed for all convolution in a block)
        N: no. of convolutional layers
    Returns:
        A sequential container of N convolutional layers.
    """
    # a single convolution + batch normalization + ReLU block
    block = lambda in_channels: nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=channels[1],
                  kernel_size=size,
                  stride=stride,
                  bias=False,
                  padding=(size[0] // 2, size[1] // 2)),
        nn.BatchNorm2d(num_features=channels[1]),
        nn.ReLU()
    )
    # create and return a sequential container of convolutional layers
    # input size = channels[0] for first block and channels[1] for all others
    return nn.Sequential(*[block(channels[bool(i)]) for i in range(N)])


class ConvCat(nn.Module):
    """Convolution with upsampling + concatenate block."""

    def __init__(self,
                 channels: Tuple[int, int],
                 size: Tuple[int, int],
                 stride: Tuple[int, int]=(1, 1),
                 N: int=1):
        """
        Create a sequential container with convolutional block (see conv_block)
        with N convolutional layers and upsampling by factor 2.
        """
        super(ConvCat, self).__init__()
        self.conv = nn.Sequential(
            conv_block(channels, size, stride, N),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, to_conv: torch.Tensor, to_cat: torch.Tensor):
        """Forward pass.
        Args:
            to_conv: input passed to convolutional block and upsampling
            to_cat: input concatenated with the output of a conv block
        """
        return torch.cat([self.conv(to_conv), to_cat], dim=1)

# from https://github.com/NeuroSYS-pl/objects_counting_dmap
# Implementations by NeuroSys Poland
class FCRN_A(nn.Module):
    """
    Fully Convolutional Regression Network A
    Ref. W. Xie et al. 'Microscopy Cell Counting with Fully Convolutional
    Regression Networks'
    """

    def __init__(self,modelname, N: int=1, input_filters: int=3, **kwargs):
        """
        Create FCRN-A model with:
            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * no. of filters as defined in an original model:
              input size -> 32 -> 64 -> 128 -> 512 -> 128 -> 64 -> 1
        Args:
            N: no. of convolutional layers per block (see conv_block)
            input_filters: no. of input channels
        """
       
        # these attr's are needed to make the model object independant of the config file
        self.dlr_acd = a.args.data == 'dlr'
        self.modelname = modelname
        self.unconditional = False
        self.count = False
        self.subnet_type = None
        self.mnist = False
        self.gap = c.gap
        self.n_coupling_blocks = 0
        self.joint_optim = False
        self.pretrained = False
        self.finetuned = False
        self.scheduler = a.args.scheduler
        self.scale = c.scale
        self.density_map_h = c.density_map_h
        self.density_map_w = c.density_map_w
        self.downsampling = c.downsampling
        self.scale = c.scale
        self.noise = a.args.noise
        self.seed = c.seed
        self.dmap_scaling = a.args.dmap_scaling
        self.sigma = a.args.sigma
        
        super(FCRN_A, self).__init__()
        self.model = nn.Sequential(
            # downsampling
            conv_block(channels=(input_filters, 32), size=(3, 3), N=N),
            nn.MaxPool2d(2),

            conv_block(channels=(32, 64), size=(3, 3), N=N),
            nn.MaxPool2d(2),

            conv_block(channels=(64, 128), size=(3, 3), N=N),
            nn.MaxPool2d(2),

            # "convolutional fully connected"
            conv_block(channels=(128, 512), size=(3, 3), N=N),

            # upsampling
            nn.Upsample(scale_factor=2),
            conv_block(channels=(512, 128), size=(3, 3), N=N),

            nn.Upsample(scale_factor=2),
            conv_block(channels=(128, 64), size=(3, 3), N=N),

            nn.Upsample(scale_factor=2),
            conv_block(channels=(64, 1), size=(3, 3), N=N),
        )

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        return self.model(input)

# implementation from: https://github.com/milesial/Pytorch-UNet
# density modification from: https://github.com/NeuroSYS-pl/objects_counting_dmap 
class UNet(nn.Module):
    """
    U-Net implementation.
    Ref. O. Ronneberger et al. "U-net: Convolutional networks for biomedical
    image segmentation."
    """

    def __init__(self, modelname, n_channels = 3, bilinear=False,seg=False,dims_out=1,subnet=False):
        """
        Create U-Net model with:
            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * fixed no. of convolutional layers per block = 2 (see conv_block)
            * constant no. of filters for convolutional layers
        Args:
            filters: no. of filters for convolutional layers
            input_filters: no. of input channels
        """
        
        
        # these attr's are needed to make the model object independant of the config file
        self.dlr_acd = a.args.data == 'dlr'
        self.modelname = modelname
        self.unconditional = False
        self.count = False
        self.subnet_type = None
        self.mnist = False
        self.gap = c.gap
        self.n_coupling_blocks = 0
        self.joint_optim = False
        self.pretrained = False
        self.finetuned = False
        self.scheduler = a.args.scheduler
        self.scale = c.scale
        self.density_map_h = c.density_map_h
        self.density_map_w = c.density_map_w
        self.downsampling = c.downsampling
        self.scale = c.scale
        self.sigma = a.args.sigma
        self.noise = a.args.noise
        self.seed = c.seed
        self.dmap_scaling = a.args.dmap_scaling
        self.seg = seg
        
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.subnet = subnet

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        if self.seg:
            self.outc = nn.Conv2d(64, 1, kernel_size=1)
        else:
            # adapt for density map estimation
            # https://github.com/NeuroSYS-pl/objects_counting_dmap/blob/master/model.py
            self.density_pred = nn.Conv2d(in_channels=64, out_channels=dims_out,
                                          kernel_size=(1, 1), bias=False)

    def forward(self, x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        if hasattr(self,'subnet') and self.subnet:
            x7 = self.up2(x4,x3) 
        else:
            x5 = self.down4(x4)
            x6 = self.up1(x5, x4)
            x7 = self.up2(x6, x3)
            
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
    
        if self.seg:
            x10 = self.outc(x9)
        else:
            x10 = self.density_pred(x9)
        
        return x10

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        network_dict = collections.OrderedDict(
                    [
                        ('conv1',nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)),
                        ('batchnorm1',nn.BatchNorm2d(mid_channels)),
                        ("relu1", nn.ReLU(inplace=True)),
                        ("conv2",nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)),
                        ('batchnorm2',nn.BatchNorm2d(out_channels)),
                        ("relu2", nn.ReLU(inplace=True))
                   
                    ]
            )
        
        if not c.batchnorm:
            del network_dict['batchnorm1']
            del network_dict['batchnorm2']
        
        self.double_conv = nn.Sequential(network_dict)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)

# https://github.com/leeyeehoo/CSRNet-pytorch- official CSRNet repo

class CSRNet(nn.Module):
    def __init__(self,modelname,load_weights=False):
        super(CSRNet, self).__init__()
        
        # these attr's are needed to make the model object independant of the config file
        self.dlr_acd = a.args.data == 'dlr'
        self.modelname = modelname
        self.unconditional = False
        self.count = False
        self.subnet_type = None
        self.mnist = False
        self.gap = c.gap
        self.n_coupling_blocks = 0
        self.joint_optim = False
        self.pretrained = False
        self.finetuned = False
        self.scheduler = a.args.scheduler
        self.scale = c.scale
        self.density_map_h = c.density_map_h
        self.density_map_w = c.density_map_w
        self.downsampling = c.downsampling
        self.scale = c.scale
        self.noise = a.args.noise
        self.seed = c.seed
        self.dmap_scaling = a.args.dmap_scaling
        self.sigma = a.args.sigma
        
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
                
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.upsample(x,scale_factor=8)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  

# LC-FCN
# From the 'Deep Fish' GitHub repository
# https://github.com/alzayats/DeepFish/blob/master/src/models/lcfcn.py
class LCFCN(nn.Module):
    def __init__(self,modelname):
        super().__init__()
        self.n_classes = 1
        
        # these attr's are needed to make the model object independant of the config file
        self.dlr_acd = a.args.data == 'dlr'
        self.modelname = modelname
        self.unconditional = False
        self.count = False
        self.subnet_type = None
        self.mnist = False
        self.gap = c.gap
        self.n_coupling_blocks = 0
        self.joint_optim = False
        self.pretrained = False
        self.finetuned = False
        self.scheduler = a.args.scheduler
        self.scale = c.scale
        self.density_map_h = c.density_map_h
        self.density_map_w = c.density_map_w
        self.downsampling = c.downsampling
        self.scale = c.scale
        self.noise = a.args.noise
        self.seed = c.seed
        self.dmap_scaling = a.args.dmap_scaling
        self.sigma = a.args.sigma
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = torchvision.models.resnet50(pretrained=True)
        resnet_block_expansion_rate = resnet50_32s.layer1[0].expansion

        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()

        self.resnet50_32s = resnet50_32s

        self.score_32s = nn.Conv2d(512 * resnet_block_expansion_rate,
                                   self.n_classes,
                                   kernel_size=1)

        self.score_16s = nn.Conv2d(256 * resnet_block_expansion_rate,
                                   self.n_classes,
                                   kernel_size=1)

        self.score_8s = nn.Conv2d(128 * resnet_block_expansion_rate,
                                  self.n_classes,
                                  kernel_size=1)


        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    @torch.no_grad()
    def predict(self, batch, **options):
        self.eval()

        if options["method"] == "counts":
            images = batch["images"].cuda()
            pred_mask = self(images).data.max(1)[1].squeeze().cpu().numpy()

            counts = np.zeros(self.n_classes - 1)

            for category_id in np.unique(pred_mask):
                if category_id == 0:
                    continue
                blobs_category = morph.label(pred_mask == category_id)
                n_blobs = (np.unique(blobs_category) != 0).sum()
                counts[category_id - 1] = n_blobs

            return counts[None]

        elif options["method"] == "blobs":
            images = batch["images"].cuda()
            pred_mask = self(images).data.max(1)[1].squeeze().cpu().numpy()

            h, w = pred_mask.shape
            blobs = np.zeros((self.n_classes - 1, h, w), int)

            for category_id in np.unique(pred_mask):
                if category_id == 0:
                    continue
                blobs[category_id - 1] = morph.label(pred_mask == category_id)

            return blobs[None]

        elif options["method"] == "points":
            images = batch["images"].cuda()
            pred_mask = self(images).data.max(1)[1].squeeze().cpu().numpy()

            h, w = pred_mask.shape
            blobs = np.zeros((self.n_classes - 1, h, w), int)

            for category_id in np.unique(pred_mask):
                if category_id == 0:
                    continue
                blobs[category_id - 1] = morph.label(pred_mask == category_id)

            return blobs[None]

    def forward(self, x):
        self.resnet50_32s.eval()
        input_spatial_dim = x.size()[2:]

        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)

        x = self.resnet50_32s.layer2(x)
        logits_8s = self.score_8s(x)

        x = self.resnet50_32s.layer3(x)
        logits_16s = self.score_16s(x)

        x = self.resnet50_32s.layer4(x)
        logits_32s = self.score_32s(x)

        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]

        logits_16s += nn.functional.interpolate(logits_32s,
                                                             size=logits_16s_spatial_dim,
                                                             mode="bilinear",
                                                             align_corners=True)

        logits_8s += nn.functional.interpolate(logits_16s,
                                                            size=logits_8s_spatial_dim,
                                                            mode="bilinear",
                                                            align_corners=True)

        logits_upsampled = nn.functional.interpolate(logits_8s,
                                                                  size=input_spatial_dim,
                                                                  mode="bilinear",
                                                                  align_corners=True)

        return logits_upsampled
  
# from milesial UNet PyTorch implementation
# https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py
def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        #inter = torch.dot(input.flatten(), target.flatten())
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

# from https://github.com/gjy3035/C-3-Framework
class MCNN(nn.Module):
    '''
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, modelname,dims_in=3,dims_out=1): # bn=False
        
        super(MCNN, self).__init__()
        
        self.dlr_acd = a.args.data == 'dlr'
        self.modelname = modelname
        self.unconditional = False
        self.count = False
        self.subnet_type = None
        self.mnist = False
        self.gap = c.gap
        self.n_coupling_blocks = 0
        self.joint_optim = False
        self.pretrained = False
        self.finetuned = False
        self.scheduler = a.args.scheduler
        self.scale = c.scale
        self.density_map_h = c.density_map_h
        self.density_map_w = c.density_map_w
        self.downsampling = c.downsampling
        self.scale = c.scale
        self.noise = a.args.noise
        self.seed = c.seed
        self.dmap_scaling = a.args.dmap_scaling
        self.sigma = a.args.sigma
        
        # todo: re add option for batch norm
        self.branch1 = nn.Sequential(nn.Conv2d( dims_in, 16, 9, padding='same'),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(16, 32, 7, padding='same'),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(32, 16, 7, padding='same'),
                                     nn.Conv2d(16,  8, 6, padding='same'))
        
        self.branch2 = nn.Sequential(nn.Conv2d( dims_in, 20, 7, padding='same'),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(20, 40, 5, padding='same'),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(40, 20, 5, padding='same'),
                                     nn.Conv2d(20, 10, 5, padding='same'))
        
        self.branch3 = nn.Sequential(nn.Conv2d( dims_in, 24, 5, padding='same'),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(24, 48, 3, padding='same'),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(48, 24, 3, padding='same'),
                                     nn.Conv2d(24, 12, 3, padding='same'))
        
        self.fuse = nn.Sequential(nn.Conv2d( 30, dims_out, 1, padding='same'))

        initialize_weights(self.modules())   
        
    def forward(self, im_data):
        
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        
        x = torch.cat((x1,x2,x3),1)
        x = self.fuse(x)
        
        # TODO: ensure this only triggers when MCNN is used as a subnet
        
        # really crappy hack to make dimensionality match feature pyramid
        # only applied to make MCNN work on 800x600 data
        # if x.size()[3] == 12:
        #     x = F.upsample(x,size=(38,50))
        # elif x.size()[3] == 6:
        #     x = F.upsample(x,size=(19,25))
        # else:
        x = F.upsample(x,scale_factor=4)
        
        return x
    
def initialize_weights(models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):    
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print( m )

def weights_normal_init(*models):
    for model in models:
        dev=0.01
        if isinstance(model, list):
            for m in model:
                weights_normal_init(m, dev)
        else:
            for m in model.modules():            
                if isinstance(m, nn.Conv2d):        
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)
                    
# ResNet 50 from:
# https://github.com/gjy3035/C-3-Framework/blob/f8e21faf8942dc852c8ed051a506789509255dc4/models/SCC_Model/Res50.py
class Res50(nn.Module):
    def __init__(self,  modelname, pretrained=True):
        super(Res50, self).__init__()
        
        self.dlr_acd = a.args.data == 'dlr'
        self.modelname = modelname
        self.unconditional = False
        self.count = False
        self.subnet_type = None
        self.mnist = False
        self.gap = c.gap
        self.n_coupling_blocks = 0
        self.joint_optim = False
        self.pretrained = False
        self.finetuned = False
        self.scheduler = a.args.scheduler
        self.scale = c.scale
        self.density_map_h = c.density_map_h
        self.density_map_w = c.density_map_w
        self.downsampling = c.downsampling
        self.scale = c.scale
        self.noise = a.args.noise
        self.seed = c.seed
        self.dmap_scaling = a.args.dmap_scaling
        self.sigma = a.args.sigma

        self.de_pred = nn.Sequential(nn.Conv2d(1024, 128, 1, padding='same'),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 1, 1, padding='same'),
                                     nn.ReLU())

        initialize_weights(self.modules())

        res = models.resnet50(pretrained=pretrained)
        # pre_wts = torch.load(model_path)
        # res.load_state_dict(pre_wts)
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)        
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

    def forward(self,x):

        
        x = self.frontend(x)

        x = self.own_reslayer_3(x)

        x = self.de_pred(x)

        x = F.upsample(x,scale_factor=8)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.data.fill_(0)   


def make_res_layer(block, planes, blocks, stride=1):

    downsample = None
    inplanes=512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)  


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out 