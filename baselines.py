# from https://github.com/NeuroSYS-pl/objects_counting_dmap
# Implementations by NeuroSys Poland
"""The implementation of U-Net and FCRN-A models."""
from typing import Tuple

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import models
from skimage import morphology as morph
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage
import torch.nn.functional as F
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


class UNet(nn.Module):
    """
    U-Net implementation.
    Ref. O. Ronneberger et al. "U-net: Convolutional networks for biomedical
    image segmentation."
    """

    def __init__(self,modelname,filters: int=64, input_filters: int=3, **kwargs):
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
        
        if a.args.dlr_acd:
            self.dlr_acd = True
        else:
            self.dlr_acd = False
        
        # these attr's are needed to make the model object independant of the config file
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
        
        super(UNet, self).__init__()
        # first block channels size
        initial_filters = (input_filters, filters)
        # channels size for downsampling
        down_filters = (filters, filters)
        # channels size for upsampling (input doubled because of concatenate)
        up_filters = (2 * filters, filters)

        # downsampling
        self.block1 = conv_block(channels=initial_filters, size=(3, 3), N=2)
        self.block2 = conv_block(channels=down_filters, size=(3, 3), N=2)
        self.block3 = conv_block(channels=down_filters, size=(3, 3), N=2)

        # upsampling
        self.block4 = ConvCat(channels=down_filters, size=(3, 3), N=2)
        self.block5 = ConvCat(channels=up_filters, size=(3, 3), N=2)
        self.block6 = ConvCat(channels=up_filters, size=(3, 3), N=2)

        # density prediction
        self.block7 = conv_block(channels=up_filters, size=(3, 3), N=2)
        self.density_pred = nn.Conv2d(in_channels=filters, out_channels=1,
                                      kernel_size=(1, 1), bias=False)

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        # use the same max pooling kernel size (2, 2) across the network
        pool = nn.MaxPool2d(2)

        # downsampling
        block1 = self.block1(input)
        pool1 = pool(block1)
        block2 = self.block2(pool1)
        pool2 = pool(block2)
        block3 = self.block3(pool2)
        pool3 = pool(block3)

        # upsampling
        block4 = self.block4(pool3, block3)
        block5 = self.block5(block4, block2)
        block6 = self.block6(block5, block1)

        # density prediction
        block7 = self.block7(block6)
        return self.density_pred(block7)
    
# https://github.com/leeyeehoo/CSRNet-pytorch- official CSRNet repo

class CSRNet(nn.Module):
    def __init__(self,modelname,load_weights=False):
        super(CSRNet, self).__init__()
        
        # these attr's are needed to make the model object independant of the config file
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
    def __init__(self):
        super().__init__()
        self.n_classes = 2

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


    def compute_loss(self, batch, i=None, ac_stp=None):
        self.train()

        images = batch["images"].cuda()
        counts = batch["counts"].cuda()

        blobs = self.predict(batch, method="blobs").squeeze()

        # self.opt.zero_grad()
        points = batch["points"].long().cuda()

        points_numpy = points.detach().cpu().numpy().squeeze()
        blob_dict = get_blob_dict_base(self, blobs, points_numpy)

        self.train()
        loss = lc_loss_base(self, images, points,
                                    counts[None], blob_dict)

        return loss

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


# ==========================================================
# Losses
def lc_loss(model, batch):
    model.train()

    blob_dict = get_blob_dict(model, batch)
    # put variables in cuda
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    return lc_loss_base(model, images, points, counts, blob_dict)
    # print(images.shape)


def lc_loss_base(logits, images, points, counts, blob_dict):
    N = images.size(0)
    assert N == 1

    S = F.softmax(logits, 1)
    S_log = F.log_softmax(logits, 1)

    # IMAGE LOSS
    loss = compute_image_loss(S, counts)

    # POINT LOSS
    loss += F.nll_loss(S_log, points,
                       ignore_index=0,
                       reduction='sum')
    # FP loss
    if blob_dict["n_fp"] > 0:
        loss += compute_fp_loss(S_log, blob_dict)

    # split_mode loss
    if blob_dict["n_multi"] > 0:
        loss += compute_split_loss(S_log, S, points, blob_dict)

    # Global loss
    S_npy = S.detach().cpu().numpy().squeeze()
    points_npy = points.detach().cpu().numpy().squeeze()
    for l in range(1, S.shape[1]):
        points_class = (points_npy == l).astype(int)

        if points_class.sum() == 0:
            continue

        T = watersplit(S_npy[l], points_class)
        T = 1 - T
        scale = float(counts.sum())
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                                          ignore_index=1, reduction='mean')

    return loss / N


# Loss Utils
def compute_image_loss(S, Counts):
    n, k, h, w = S.size()

    # GET TARGET
    ones = torch.ones(Counts.size(0), 1).long().cuda()
    BgFgCounts = torch.cat([ones.float(), Counts.float()], 1)
    Target = (BgFgCounts.view(n * k).view(-1) > 0).view(-1).float()

    # GET INPUT
    Smax = S.view(n, k, h * w).max(2)[0].view(-1)

    loss = F.binary_cross_entropy(Smax, Target, reduction='sum')

    return loss


def compute_fp_loss(S_log, blob_dict):
    blobs = blob_dict["blobs"]

    scale = 1.
    loss = 0.
    n_terms = 0
    for b in blob_dict["blobList"]:
        if n_terms > 25:
            break

        if b["n_points"] != 0:
            continue

        T = np.ones(blobs.shape[-2:])
        T[blobs[b["class"]] == b["label"]] = 0

        loss += scale * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                                   ignore_index=1, reduction='mean')

        n_terms += 1
    return loss


def compute_bg_loss(S_log, bg_mask):
    loss = F.nll_loss(S_log, torch.LongTensor(bg_mask).cuda()[None],
                      ignore_index=1, reduction='mean')
    return loss


def compute_split_loss(S_log, S, points, blob_dict):
    blobs = blob_dict["blobs"]
    S_numpy = S[0].detach().cpu().numpy()
    points_numpy = points.detach().cpu().numpy().squeeze()

    loss = 0.

    for b in blob_dict["blobList"]:
        if b["n_points"] < 2:
            continue

        l = b["class"] + 1
        probs = S_numpy[b["class"] + 1]

        points_class = (points_numpy == l).astype("int")
        blob_ind = blobs[b["class"]] == b["label"]

        T = watersplit(probs, points_class * blob_ind) * blob_ind
        T = 1 - T

        scale = b["n_points"] + 1
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                                          ignore_index=1, reduction='mean')

    return loss


def watersplit(_probs, _points):
    points = _points.copy()

    points[points != 0] = np.arange(1, points.sum() + 1)
    points = points.astype(float)

    probs = ndimage.black_tophat(_probs.copy(), 7)
    seg = watershed(probs, points)

    return find_boundaries(seg)


@torch.no_grad()
def get_blob_dict(model, batch, training=False):
    blobs = model.predict(batch, method="blobs").squeeze()
    points = batch["points"].detach().cpu().numpy().squeeze()

    return get_blob_dict_base(model, blobs, points, training=training)


def get_blob_dict_base(model, blobs, points, training=False):
    if blobs.ndim == 2:
        blobs = blobs[None]

    blobList = []

    n_multi = 0
    n_single = 0
    n_fp = 0
    total_size = 0

    for l in range(blobs.shape[0]):
        class_blobs = blobs[l]
        points_mask = points == (l + 1)
        # Intersecting
        blob_uniques, blob_counts = np.unique(class_blobs * (points_mask), return_counts=True)
        uniques = np.delete(np.unique(class_blobs), blob_uniques)

        for u in uniques:
            blobList += [{"class": l, "label": u, "n_points": 0, "size": 0,
                          "pointsList": []}]
            n_fp += 1

        for i, u in enumerate(blob_uniques):
            if u == 0:
                continue

            pointsList = []
            blob_ind = class_blobs == u

            locs = np.where(blob_ind * (points_mask))

            for j in range(locs[0].shape[0]):
                pointsList += [{"y": locs[0][j], "x": locs[1][j]}]

            assert len(pointsList) == blob_counts[i]

            if blob_counts[i] == 1:
                n_single += 1

            else:
                n_multi += 1
            size = blob_ind.sum()
            total_size += size
            blobList += [{"class": l, "size": size,
                          "label": u, "n_points": blob_counts[i],
                          "pointsList": pointsList}]

    blob_dict = {"blobs": blobs, "blobList": blobList,
                 "n_fp": n_fp,
                 "n_single": n_single,
                 "n_multi": n_multi,
                 "total_size": total_size}

    return blob_dict