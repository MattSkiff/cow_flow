# we are following the YOLO format for data formatting and bounding box annotations
# see here: https://github.com/AlexeyAB/Yolo_mark/issues/60 for a description

# edit tutorial to work for object annotations and YOLO format
import os 
import torch 
import pandas as pd
import numpy as np
import random
import time
import cv2 # imread
from tqdm import tqdm # progress bar
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# silent warning on (torch.stack(images))
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

# https://docs.opencv.org/4.5.2/d4/d13/tutorial_py_filtering.html
import scipy
#from skimage import restoration
#from scipy.signal import convolve2d as conv2

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF
import torchvision.transforms as T
#import torch.nn.functional as F

from torchvision import utils
from skimage import io
from PIL import Image

import config as c
import gvars as g
import arguments as a

from utils import UnNormalize
from utils import split_image

# save numpy array as npz file
from numpy import savez_compressed, load

proj_dir = c.proj_dir
random_flag = False
points_flag = True
demo = False
density_demo = True
mk_size = 25

#### DLR ACD classes 

class DLRACD(Dataset):
    
    # TODO add image name so concordance with gsd table is possible
    def __init__(self, root_dir,overlap=50,transform = None):
        self.root_dir = root_dir + '/DLRACD/'
        
        # this dict structure is only used for processing in init method,
        # switch to flat list at end
        self.count = False
        self.dlr_acd = True
        self.classification = False # TODO - no null filtering - yet
        self.transform = transform
        self.overlap = overlap
        
        self.images = {'Train':[],'Test':[]}
        
        self.counts = {'Train':[],'Test':[]}
        self.density_counts = {'Train':[],'Test':[]}
        
        self.annotations = {'Train':[]}
        self.anno_path = {'Train':[]}
        
        self.train_indices = []
        self.test_indices = []
        
        self.patches = {'Train':[],'Test':[]}
        self.patch_densities = {'Train':[],'Test':[]}
        self.patch_path = {'Train':[],'Test':[]}
        
        self.point_maps = {'Train':[]}
        self.point_map_path = {'Train':[]}
        
        gsd_table_df = pd.read_csv(self.root_dir+"dlracd_gsds.txt")
        
        print('Initialising dataset...')
        
        ### Patches section
        for mode in ['Test','Train']:
            
            i_paths  = os.listdir(os.path.join(self.root_dir,mode+"/Images/"))
            image_files = [f for f in i_paths if os.path.isfile(os.path.join(self.root_dir,mode+"/Images/", f))]
            images_list = sorted(image_files)
            
            if overlap == 0:
                
                for img_path in tqdm(images_list, desc="Loading and splitting {} images...".format(mode)):
                    
                    im = cv2.imread(os.path.join(self.root_dir,mode+"/Images/"+img_path))
                    self.images[mode].append(im)
                    im_patches = split_image(im, patch_size = 320,save = False, overlap=0)
                    self.patches[mode].extend(im_patches)
                    self.patch_path[mode].extend([img_path]*len(im_patches))
            
            else:
                
                patch_save = not os.path.exists(self.root_dir+mode+"/Images/Patches/")  
                im_patch_path = self.root_dir+mode+"/Images/Patches/"
                
                if patch_save:
                    os.makedirs(im_patch_path, exist_ok = False)
                    
                # create image patches
                for img_path in tqdm(images_list, desc="Loading and splitting {} images...".format(mode)):
                    im = cv2.imread(os.path.join(self.root_dir,mode+"/Images/"+img_path))
                    im_patches = split_image(im, save = patch_save ,patch_size = 320, overlap=overlap,name = img_path[:-4],path = im_patch_path,frmt = 'jpg')
        
                patches_list = sorted(os.listdir(os.path.join(self.root_dir,mode+"/Images/Patches/")))
                
                for patch_path in tqdm(patches_list, desc="Loading {} patches...".format(mode)):
                    
                    im_patch = cv2.imread(os.path.join(self.root_dir,mode+"/Images/Patches/"+patch_path))
                    self.patches[mode].append(im_patch) # (320, 320, 3) 
                    self.patch_path[mode].append(os.path.join(self.root_dir,mode+"/Images/Patches/"+patch_path))
            
            if mode == 'Train': # only Train data available
            
                ### Annotations section
                a_paths  = os.listdir(os.path.join(self.root_dir,mode+"/Annotation/"))
                anno_files = [f for f in a_paths if os.path.isfile(os.path.join(self.root_dir,mode+"/Annotation/", f))]
                annotation_list = sorted(anno_files)
                
                if overlap == 0:
    
                    for anno_path in tqdm(annotation_list, desc="Loading and splitting annotations..."):
                        
                        # read in as greyscale (0 arg)
                        annotation = cv2.imread(os.path.join(self.root_dir,mode+"/Annotation/"+anno_path),0)
                        self.annotations[mode].append(annotation)
                        point_maps = split_image(annotation, patch_size = 320,save = False, overlap=0)   
                        
                        for pm in point_maps:
                            
                            pm = pm / 255
                            self.counts[mode].append(pm.sum())
                            self.point_maps[mode].append(pm)
                            self.point_map_path[mode].append(anno_path)
                            
                            # NB convert pd series to list, 0th element                   
                            gsd_sigma = gsd_table_df[gsd_table_df['name'] == anno_path[:-4]]['gsd'].values[0]
                            
                            if c.debug:
                                print('Anno path is {}'.format(anno_path))
                                print('GSD is {}'.format(gsd_sigma))
                            
                            if a.args.dmap_type == 'max':
                                p_d = scipy.ndimage.filters.maximum_filter(pm,size = (7,7))
                            else:
                                p_d = scipy.ndimage.filters.gaussian_filter(pm, sigma = gsd_sigma, mode='constant')
                                
                            self.density_counts[mode].append(p_d.sum())
                            self.patch_densities[mode].append(p_d)
                            
                else:
                    
                    patch_save = not os.path.exists(self.root_dir+mode+"/Annotation/Patches/") 
                    anno_patch_path = self.root_dir+mode+"/Annotation/Patches/"
                    
                    if patch_save:
                        os.makedirs(anno_patch_path, exist_ok = False)
                    
                    # create annotation patches
                    for anno_path in tqdm(annotation_list, desc="Loading and splitting images..."):
                        anno = cv2.imread(os.path.join(self.root_dir,mode+"/Annotation/"+anno_path))
                        anno_patches = split_image(anno,save=patch_save, patch_size = 320, overlap=overlap,name = anno_path[:-4],path = anno_patch_path,frmt = 'png')
                        self.anno_path[mode].extend([anno_path]*len(anno_patches))
                    
                    anno_patches_list = sorted(os.listdir(os.path.join(self.root_dir,mode+"/Annotation/Patches/")))
                    
                    for anno_patch_path,anno_path in tqdm(zip(anno_patches_list,self.anno_path[mode]), desc="Loading annotation patches..."):
                        
                        # read in as greyscale (0 arg)
                        pm = cv2.imread(os.path.join(self.root_dir,mode+"/Annotation/Patches/"+anno_patch_path),0) 
                        pm = pm / 255
                        
                        self.counts[mode].append(pm.sum())
                        self.point_maps[mode].append(pm)
                        self.point_map_path[mode].append(anno_patch_path)
                        
                        # NB convert pd series to list, 0th element                   
                        gsd_sigma = gsd_table_df[gsd_table_df['name'] == anno_path[:-4]]['gsd'].values[0]
                        
                        if c.debug:
                            print('Anno path is {}'.format(anno_path))
                            print('Anno patch path is {}'.format(anno_patch_path))
                            print('GSD is {}'.format(gsd_sigma))
                        
                        p_d = scipy.ndimage.filters.gaussian_filter(pm, sigma = gsd_sigma, mode='constant')
                        self.density_counts[mode].append(p_d.sum())
                        self.patch_densities[mode].append(p_d)
                        
        
        ## jointly shuffle data indices  
        joint_lists = list(zip(self.patches['Train'],self.patch_densities['Train'],
                              self.counts['Train'],self.density_counts['Train'],self.point_maps['Train']))
        
        random.shuffle(joint_lists)
        self.patches['Train'],self.patch_densities['Train'],self.counts['Train'],self.density_counts['Train'],self.point_maps['Train'] = zip(*joint_lists)
        
        break_point = round(0.7*len(self.patches['Train']))
        self.train_indices = range(0,break_point)
        self.test_indices = range(break_point,len(self.patches['Train']))
        #self.test_indices =  range(len(self.train_indices),len(self.train_indices)+len(self.patches['Test']))
        
        self.patches = [*self.patches['Train'],*self.patches['Test']]
        self.patch_path = [*self.patch_path['Train'],*self.patch_path['Test']]
        self.patch_densities = [*self.patch_densities['Train']] # no test annotations provided
        self.counts = [*self.counts['Train']]
        self.density_counts = [*self.density_counts['Train']]
        self.point_maps = [*self.point_maps['Train']]
        self.point_map_path = [*self.point_map_path['Train']]
        
        self.hparam_dict = None
        self.metric_dict = None
        
        
        print("Number of DLR ACD image patches: {}".format(len(self.patches)))
        print("Number of DLR ACD point map patches: {}".format(len(self.point_maps)))
        print("Number of DLR ACD density map patches: {}".format(len(self.patch_densities)))
        time.sleep(1)
        
        print('\nInitialisation finished')
                    
    def __len__(self):
        return len(self.patches['Train'])+len(self.patches['Test'])
        
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        sample['patch'] = self.patches[idx]
        
        if idx < len(self.patch_densities):
            
            # if self.overlap == 0:
            sample['patch_density'] = self.patch_densities[idx]
            sample['counts'] = self.counts[idx]
            sample['density_counts'] = self.density_counts[idx]
            sample['point_maps'] = self.point_maps[idx] 
        
        else:
            sample['patch_density'] = None
            sample['counts'] = None
            sample['density_counts'] = None
            sample['point_maps'] = None
            
        if self.transform:
             sample = self.transform(sample)
            
        return sample
    
    def show_annotations(self, idx):
        
        sample = self[idx]
        
        print("True Count:")
        print(sample["counts"])
        print('Paths')
        print((self.patch_path[idx]+" | "+self.point_map_path[idx]))
        
        patch = sample['patch'].permute((1, 2, 0)).cpu().numpy()
        point_map = sample['point_maps'].cpu().numpy()
        gt_coords = np.argwhere(point_map == 1)
        patch_densities = sample['patch_density'].cpu().numpy()

        fig, ax = plt.subplots(1,3,figsize=(24, 8))
        
        ax[0].imshow(patch_densities, cmap='viridis', interpolation='nearest', aspect="auto")
        ax[1].imshow((patch * 255).astype(np.uint8), aspect="auto")
        ax[2].imshow((point_map * 255).astype(np.uint8), aspect="auto")
        if len(gt_coords) != 0:
            ax[2].scatter(gt_coords[:,1], gt_coords[:,0],c='red',marker='1',s=30,
                          label='Ground truth coordinates') 
            
        #ax[2].set_ylim([0, 320])
        #ax[2].set_xlim([0, 320])
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.90)
        fig.suptitle('Sample {}: \nDensity Count: {}, True Count: {}'.format(idx,round(sample['density_counts'],2),sample['counts']),y=1.0,fontsize=24)
        
    def custom_collate_aerial(self,batch):
        
        patches = list()
        patch_densities = list()
        counts = list()
        point_maps = list()
        
        for b in batch:
                  
            patches.append(b['patch'])
            patch_densities.append(b['patch_density'])
            counts.append(b['counts']) 
            point_maps.append(b['point_maps']) 

        patches = torch.stack(patches,dim = 0)
        patch_densities = torch.stack(patch_densities,dim = 0)
        counts = torch.stack(counts,dim = 0)
        point_maps = torch.stack(point_maps,dim = 0)
        
        if c.debug:
            print(type(patches))
            print(type(patch_densities))
            print(type(counts))
        
        collated_batch = patches,patch_densities,counts,point_maps
        
        return collated_batch

class DLRACDToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        patch = sample['patch']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        patch = torch.from_numpy(patch)
        patch = patch.permute(2,0,1)

        if c.debug:
            print("patch transposed from numpy format")
            print("patch rescaled to 0-1 range")
            
        patch = patch.float().div(255).to(c.device)
        
        sample['patch'] =  patch.float()
        
        if 'patch_density' in sample.keys():
            sample['patch_density'] = torch.from_numpy(sample['patch_density']).to(c.device)
        if 'counts' in sample.keys():
            sample['counts'] = torch.from_numpy(np.array(sample['counts']).astype(float)).to(c.device)
        if 'point_maps' in sample.keys():
            sample['point_maps'] = torch.from_numpy(sample['point_maps']).to(c.device)

        return sample
    
class DLRACDAddUniformNoise(object):
    """Add uniform noise to Dmaps to stabilise training."""
    def __init__(self, r1=0., r2=a.args.noise):
        self.r1 = r1
        self.r2 = r2
    
    def __call__(self, sample):
        # uniform tensor in pytorch: 
        # https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        
        if 'patch_density' in sample.keys():
            pass
            sample['patch_density'] =  sample['patch_density'] + torch.FloatTensor(sample['patch_density'].size()).uniform_(self.r1, self.r2).to(c.device)
        
            if c.debug:
                print("uniform noise ({},{}) added to patch density".format(self.r1, self.r2))
        
        return sample

class DLRACDCropRotateFlipScaling(object):
    """Randomly rotate, flip and scale aerial image and density map."""

    def __call__(self, sample):
              
        # Left - Right, Up - Down flipping
        # 1/4 chance of no flip, 1/4 chance of no rotation, 1/16 chance of no flip or rotate
        resize = T.Resize(size=(256,256))
        
        sample['patch'] = resize(sample['patch'].unsqueeze(0))
        sample['patch_density'] = resize(sample['patch_density'].unsqueeze(0).unsqueeze(0))
        sample['point_maps'] = resize(sample['point_maps'].unsqueeze(0).unsqueeze(0))
        
        
        i, j, h, w = T.RandomCrop.get_params(sample['patch'], output_size=(256, 256))
        sample['patch'] = TF.crop(sample['patch'], i, j, h, w)
        sample['patch_density'] = TF.crop(sample['patch_density'], i, j, h, w)
        sample['point_maps'] = TF.crop(sample['point_maps'], i, j, h, w)
        
        if random.randint(0,1):
            sample['patch'] = torch.flip(sample['patch'],(3,))
            sample['patch_density'] = torch.flip(sample['patch_density'],(3,))
            sample['point_maps'] = torch.flip(sample['point_maps'],(3,))
            
        if random.randint(0,1):
            sample['patch'] = torch.flip(sample['patch'],(2,))
            sample['patch_density'] = torch.flip(sample['patch_density'],(2,))
            sample['point_maps'] = torch.flip(sample['point_maps'],(2,))
            
        rangle = float(random.randint(0,3)*90)
        sample['patch'] = TF.rotate(sample['patch'],angle=rangle)
        sample['patch_density'] = TF.rotate(sample['patch_density'],angle=rangle)
        sample['point_maps'] = TF.rotate(sample['point_maps'],angle=rangle)
        
        sample['patch'] = sample['patch'].squeeze()
        sample['patch_density'] = sample['patch_density'].squeeze().squeeze()
        sample['point_maps'] = sample['point_maps'].squeeze().squeeze()

        return sample
    
#### Cow Flow Classes 

# GSD = 0.3m

class CowObjectsDataset(Dataset):
    """Cow Objects dataset."""
    
    def __init__(self, root_dir,transform=None,convert_to_points=False,generate_density=False,count=False,classification=False,ram=False):
        """
        Args:
            root_dir (string): Directory with the following structure:
                object.names file
                object.data file
                obj directory containing imgs and txt annotations
                    each img has a corresponding txt file
                test.txt file
                train.txt file
                
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.points = convert_to_points
        self.density = generate_density
        self.count = count
        self.classification = classification
        self.ram = ram
        
        if not self.density:
            self.sigma = 0
        else:
            self.sigma = c.sigma
        
        self.images = []
        self.annotations_list = []
        self.density_list = []
        self.labels_list = []
        self.count_list = []
        self.binary_labels_list = []
        self.im_names = []
        
        names = []
        with open(os.path.join(self.root_dir,"object.names")) as f:
            names_input = f.readlines()
        
        for line in names_input:
            names.append(line.strip())
        
        data = {}
        with open(os.path.join(self.root_dir,"object.data")) as f:
                  data_input = f.readlines()
        
        for line in data_input:
            data[line.split("=")[0].strip()] = line.split("=")[1].strip()
        
        self.names = names
        self.data = data
        self.train_path = os.path.join(self.root_dir,os.path.split(self.data["train"])[1])
        self.test_path = os.path.join(self.root_dir,os.path.split(self.data["valid"])[1])
        
        train_im_paths = [] 
        with open(os.path.join(self.root_dir,self.train_path)) as f:
            im_names_input = f.readlines()
        
        for line in im_names_input:
            # don't start relative string with slash, or they are considered absolute 
            # and everything prior will be discarded
            # https://stackoverflow.com/questions/1945920/why-doesnt-os-path-join-work-in-this-case
            train_im_paths.append(os.path.join(self.root_dir,'obj/',line.strip()))
            self.im_names.append(line)
            
        self.train_im_paths = train_im_paths
        
        def compute_labels(idx):
                         
            if a.args.dmap_type == 'max':
                dmap_type = '_max'
            else:
                dmap_type = '' # _gauss
            
            """ computes and returns labels for a single annotation file"""
            labels = []
            
            img_path = os.path.join(self.root_dir,
                                    self.train_im_paths[idx])
            
            txt_path = img_path[:-3]+'txt'
            image = io.imread(img_path)
            
            # check if annotations file is empty
            header_list = ['class', 'x', 'y', 'width', 'height']
            
            with open(txt_path) as annotations:
                count = len(annotations.readlines())
            
            dmap_path = g.DMAP_DIR+self.im_names[idx]
            
            if c.load_stored_dmaps:
                if not os.path.exists(dmap_path):
                    ValueError("Dmaps must have been previously stored!")
   
                store = load(dmap_path[:-5]+dmap_type+'.npz',allow_pickle=True)
                density_map = store['arr_0'][0]
                labels = store['arr_0'][1]
                annotations = store['arr_0'][2]
                    
            else:
            
                if count == 0:
                        
                    annotations = np.array([])
                    density_map = np.zeros((c.img_size[1], c.img_size[0]), dtype=np.float32)
                    
                else:        
                    annotations = pd.read_csv(txt_path,names=header_list,delim_whitespace=True)
                    annotations = annotations.to_numpy()
                    
                    if self.points:
                        # convert annotations to points 
                        # delete height and width columns, which won't be used in density map
                        # becomes: <object-class> <centre-x> <centre-y>
                        annotations = np.delete(arr = annotations,obj = [3,4],axis = 1) # np 2d: row = axis 0, col = axis 1
                        
                    if self.density:
                        if not self.points:
                            print("Generation of maps requires conversion to points")
                            
                        assert self.points  
                        
                        # running gaussian filter over points as in crowdcount mcnn
                        # https://github.com/svishwa/crowdcount-mcnn/blob/master/data_preparation/get_density_map_gaussian.m
                        # https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
                        
                        # map generation
                        # set density map image size equal to data image size
                        
                        # TODO - this is possibly massively inefficient
                        density_map = np.zeros((c.img_size[1], c.img_size[0]), dtype=np.float32)
                       
                        # add points onto basemap
                        for point in annotations:
                            # error introduced here as float position annotation centre converted to int
                            base_map = np.zeros((c.img_size[1], c.img_size[0]), dtype=np.float32)
                            
                            if c.debug_dataloader:
                                print(point)
                                
                            # subtract 1 to account for 0 indexing
                            base_map[int(round(point[2]*c.img_size[1])-1),int(round(point[1]*c.img_size[0])-1)] += 1
                            
                            if a.args.dmap_type == 'max':
                                density_map += scipy.ndimage.filters.maximum_filter(base_map,size = (7,7))
                            else:                                
                                density_map += scipy.ndimage.filters.gaussian_filter(base_map, sigma = c.sigma, mode='constant')
                            
                            labels.append(point[0])
                            
                            if c.debug_dataloader:
                                print("base map sum ",base_map.sum())
                                print("density map sum ",density_map.sum())    
                                
                labels = np.array(labels) # list into default collate function produces empty tensors
                
                # store dmaps/labels/annotations
                if c.store_dmaps:
                    if not os.path.exists(g.DMAP_DIR):
                        os.makedirs(g.DMAP_DIR)
                    
                    store = [density_map,labels,annotations]
                    savez_compressed(dmap_path[:-5]+dmap_type,store)
            
            sample = {'image': image}
                
            if self.density and not self.count:
                
                # if a.args.model_name == 'CSRNet':
                #     tgt = density_map
                #     density_map = cv2.resize(tgt,(int(tgt.shape[1]/8),int(tgt.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64

                sample['density'] = density_map; sample['labels'] = labels
                
            if self.count:
                sample['counts'] = torch.as_tensor(count).float().to(c.device)
            
            if self.classification:
                positive = (len(annotations) == 0)
                sample['binary_labels'] = torch.tensor(positive).type(torch.LongTensor).to(c.device)
                
            sample['annotations'] = annotations
                
            return sample
        
        self.compute_labels = compute_labels
        
        if self.ram:

            # for img_path in tqdm(self.train_im_paths,desc="Loading aerial images into RAM"): 
                
            #     self.images.append(io.imread(img_path))
            
            if ram and c.load_stored_dmaps:
                desc = "Loading images, annotations, dmaps and labels into RAM"
            elif ram and c.store_dmaps:
                desc = "Storing dmaps, annotations and labels to file"
            elif ram:
                desc = "Computing dmaps, annotations and labels and storing into RAM"
            
            for idx in tqdm(range(len(self.train_im_paths)),desc=desc):
                
                sample = compute_labels(idx)
                
                self.images.append(sample['image'])
                                  
                if self.density and not self.count:  
                    self.density_list.append(sample['density'])
                    self.labels_list.append(sample['labels'])
                if self.count:
                    self.count_list.append(sample['counts'])
                if self.classification:
                    self.binary_labels_list.append(sample['binary_labels'])
                    
                self.annotations_list.append(sample['annotations'])
        
        
    def __len__(self):
        if self.ram:
            out = len(self.images)
        else:
            with open(os.path.join(self.root_dir,"train.txt")) as f:
                      train_ims = f.readlines()
                      
            out = len(train_ims)
        
        return out

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        # yolo object annotation columns
        # <object-class> <x> <y> <width> <height>
        if self.ram:
            
            sample['image'] = self.images[idx]
            
            if not self.density:
                sample['annotations'] = self.annotations_list[idx]
            if self.density and not self.count:  
                sample['density'] = self.density_list[idx]
                sample['labels'] = self.labels_list[idx]
            if self.count:
                sample['counts'] = self.count_list[idx]
            if self.classification:
                sample['binary_labels'] = self.binary_labels_list[idx]
                
            sample['annotations'] = self.annotations_list[idx]
                
        else:
            sample = self.compute_labels(idx)
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample

    def get_annotations(self, idx):
        """retrieve just the annotations associated with the sample"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.train_im_paths[idx])
        
        txt_path = img_path[:-3]+'txt'

        # yolo object annotation columns
        # <object-class> <x> <y> <width> <height>
        
        # check if annotations file is empty
        header_list = ['class', 'x', 'y', 'width', 'height']
        
        if os.stat(txt_path).st_size == 0:
            annotations = np.array([])
        else:        
            annotations = pd.read_csv(txt_path,names=header_list,delim_whitespace=True)
            annotations = annotations.to_numpy()
            
        return annotations
        
        
    # Helper function to show a batch
    def show_annotations_batch(self,sample_batched,debug = False):
        
        """Show image with annotations for a batch of samples."""
        
        batch_size = len(sample_batched[0])
        
        if self.density: 
            ncol = 2 
        else:
            ncol = 1
        
        fig, ax = plt.subplots(batch_size,ncol,figsize=(8,13))
        plt.ioff()
        fig.suptitle('Batch from dataloader',y=0.95,fontsize=20)
        #fig.set_size_inches(8*ncol,6*batch_size)
        #fig.set_dpi(100)
      
        images_batch = sample_batched[0]
      
        if self.density:
            
            density_batch = sample_batched[1]
            
            for i in range(batch_size):
                density = density_batch[i]
                image = images_batch[i]
                image = image.permute((1, 2, 0)).squeeze().cpu().numpy()
                ax[i,0].axis('off')
                ax[i,1].axis('off')
                ax[i,0].imshow(density, cmap='hot', interpolation='nearest')
                ax[i,1].imshow((image * 255).astype(np.uint8))
        else:
        
            annotations_batch = sample_batched[1]
            
            for i in range(batch_size):
                grid = utils.make_grid(images_batch[i])
                ax[i].imshow(grid.numpy().transpose((1, 2, 0)))
                ax[i].axis('off')
            
            if debug:
                print(annotations_batch)
        
            
            for i in range(batch_size):
                # sample_batched: image, annotations, classes
                an = sample_batched[1][i]
                if len(an) != 0:
                    an = an.numpy()
                    if not self.points:
                        for j in range(0,len(an)):
                            
                            if c.debug:
                                print(an)
                            
                            x = an[j,0]*c.img_size[0]-0.5*an[j,2]*c.img_size[0]
                            y = an[j,1]*c.img_size[1]-0.5*an[j,3]*c.img_size[1]
                            
                            rect = patches.Rectangle(xy = (x,y),
                                                 width = an[j,2]*c.img_size[0],
                                                 height = an[j,3]*c.img_size[1], 
                                                 linewidth=1, edgecolor='r', facecolor='none')
                            # Add the patch to the Axes
                            ax[i].add_patch(rect)
                    else:
                        num_cols = an.shape[1]
                        if num_cols != 2:
                            ax[i].scatter(an[:,1]*c.img_size[0],an[:,2]*c.img_size[1], s=mk_size,color = 'red') 
                        else:
                            ax[i].scatter(an[:,0]*c.img_size[0],an[:,1]*c.img_size[1], s=mk_size,color = 'red')
                                
        plt.show()
            
    # helper function to show annotations + example
    def show_annotations(self,sample_no,title = "", save = False,show = False,debug = False):
        
        sample = self[sample_no]
        
        if self.count:
            print("Count:")
            print(sample["counts"])
            return
        
        # TODO - plot annotations as well (i.e. pre dmap)
        
        if self.density:
                im = sample['image']
                dmap = sample['density'].cpu().numpy()
                
                unnorm = UnNormalize(mean =tuple(c.norm_mean),std=tuple(c.norm_std))
                im = unnorm(im)
                im = im.permute(1,2,0).cpu().numpy()
                
                fig, ax = plt.subplots(1,2)
                # TODO - bug here! mismatch impaths and data
                #fig.suptitle(os.path.basename(os.path.normpath(self.train_im_paths[sample_no])))
                ax[0].imshow(dmap, cmap='viridis', interpolation='nearest')
                ax[1].imshow((im * 255).astype(np.uint8))
        else:
            """Show image with landmarks"""
            image = sample['image']
            an = sample['annotations']
            
            fig, ax = plt.subplots(figsize=(3,4))
            ax.imshow(image)
            
            if debug:
                print(an)
            
            # define patch using x1,x2,y1,y2 coords -> map to height and width
            # <object-class> <centre-x> <centre-y> <width> <height>
            
            if len(an) != 0:
                if not self.points:
                    for i in range(0,len(an)):
                        rect = patches.Rectangle(xy = (an[i,1]*c.img_size[0]-0.5*an[i,3]*c.img_size[0],
                                                       an[i,2]*c.img_size[1]-0.5*an[i,4]*c.img_size[1]),
                                                 width = an[i,3]*c.img_size[0],height = an[i,4]*c.img_size[1], 
                                                 linewidth=1, edgecolor='r', facecolor='none')
                        # Add the patch to the Axes
                        ax.add_patch(rect)
                else:
                    num_cols = an.shape[1]
                    if num_cols != 2:
                        ax.scatter(an[:,1]*c.img_size[0],an[:,2]*c.img_size[1], s=mk_size,color = 'red') 
                    else:
                        ax.scatter(an[:,0]*c.img_size[0],an[:,1]*c.img_size[1], s=mk_size,color = 'red')
                        
        # turn off whitespace etc
        plt.axis('off')
        
        if self.density:
            for i in range(0,len(ax)):
                ax[i].get_xaxis().set_visible(False)
                ax[i].get_yaxis().set_visible(False)
        else:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
        if title != "" and self.density:
            ax[0].set_title(title)
        elif title != "":
            ax.set_title(title)
            
        if save:      
            plt.savefig(cow_dataset.root_dir+"plot.jpg", bbox_inches='tight', pad_inches = 0)
                
        if show:
            plt.show()
    
        return im, dmap
            
    # because batches have varying numbers of bounding boxes, need to define custom collate func
    # https://discuss.pytorch.org/t/dataloader-gives-stack-expects-each-tensor-to-be-equal-size-due-to-different-image-has-different-objects-number/91941/5
    
    # method to stack tensors of different sizes:
    # https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py

    def custom_collate_aerial(self,batch,debug = False):
        # only needed to collate annotations

        images = list()
        density = list()
        labels = list()
        annotations = list()
        binary_labels = list()
        
        if self.count:
            counts = list()
        
        for b in batch:
                  
            images.append(b['image'])
              
            if 'density' in b.keys():
                density.append(b['density'])
            
            if 'labels' in b.keys():
                labels.append(b['labels'])
                
            if 'annotations' in b.keys():
                annotations.append(b['annotations'])
            
            if self.count:
                counts.append(b['counts']) 
                
            if self.classification:
                 binary_labels.append(b['binary_labels'])

        images = torch.stack(images,dim = 0)
        
        if 'density' in b.keys():
            density = torch.stack(density,dim = 0)
          
        if 'labels' in b.keys():
            labels = labels
            
        if 'annotations' in b.keys():
            annotations = annotations
        
        if self.count:
            counts = torch.stack(counts,dim = 0)
            
        if self.classification:
            binary_labels = torch.stack(binary_labels,dim = 0)
        
        if debug:
            print(type(density))
            print(type(labels))
            print(type(images))
            print(density)
        
        out = images,density,labels
        
        if self.count:
            out = out + (counts,)
            
        if self.classification:
            out = out + (binary_labels,)
            
        out = out + (annotations,)
    
        return out

# Define transform to tensor
# Rescale transform not needed as slices are all same size
# RandomCrop not needed (unnecesarry for this task)
class CustToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)

        if c.debug:
            print("image transposed from numpy format")
            print("Image rescaled to 0-1 range")
        
        # since we are skipping pytorch ToTensor(), we need to manually scale to 0-1
        # RGB normalization
        # to align with transforms expected by pytorch model zoo models
        # issue: https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683/3
        # code: https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122/5
        
        # edit: normalization is v. simply - simply divide by 255
        image = image.float().div(255).to(c.device)
        
        sample['image'] =  image.float()
        
        if 'density' in sample.keys():
            sample['density'] = torch.from_numpy(sample['density']).to(c.device)
        if 'annotations' in sample.keys():
            sample['annotations'] = torch.from_numpy(sample['annotations']).to(c.device)
        if 'labels' in sample.keys():
            sample['labels'] = torch.from_numpy(sample['labels']).to(c.device)

        return sample
    
class AerialNormalize(object):
    """Call Normalize transform only on images ."""

    def __call__(self, sample):
        
        if c.debug_dataloader:
            print('aerial normalize size')
            print(sample['image'].size())
            
        sample['image'] = TF.normalize(sample['image'], 
                                       mean = c.norm_mean,
                                       std= c.norm_std)
 
        return sample

class DmapAddUniformNoise(object):
    """Add uniform noise to Dmaps to stabilise training."""
    def __init__(self, r1=0., r2=a.args.noise):
        self.r1 = r1
        self.r2 = r2
    
    def __call__(self, sample):
        # uniform tensor in pytorch: 
        # https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        
        if 'density' in sample.keys():
            pass
            sample['density'] =  sample['density'] + torch.FloatTensor(sample['density'].size()).uniform_(self.r1, self.r2).to(c.device)
        
            if c.debug:
                print("uniform noise ({},{}) added to dmap".format(self.r1, self.r2))
        
        return sample

class CropRotateFlipScaling(object):
    """Randomly rotate, flip and scale aerial image and density map."""

    def __call__(self, sample):
              
        # Left - Right, Up - Down flipping
        # 1/4 chance of no flip, 1/4 chance of no rotation, 1/16 chance of no flip or rotate
        # want identical transforms to density and image
        # https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7
        resize = T.Resize(size=(256,256))
        
        # TODO - fix this by swapping crop and resize (currently no random crop, just flips)
        
        sample['image'] = resize(sample['image'].unsqueeze(0))
        
        if not a.args.model_name == 'CSRNet':
            sample['density'] = resize(sample['density'].unsqueeze(0).unsqueeze(0))
        else:
            resize = T.Resize(size=(32,32))
            sample['density'] = resize(sample['density'].unsqueeze(0).unsqueeze(0))
            
        
        i, j, h, w = T.RandomCrop.get_params(sample['image'], output_size=(256, 256))
        sample['image'] = TF.crop(sample['image'], i, j, h, w)
        
        if not a.args.model_name == 'CSRNet':
            sample['density'] = TF.crop(sample['density'], i, j, h, w)

        if random.randint(0,1):
            sample['image'] = torch.flip(sample['image'],(3,))
            sample['density'] = torch.flip(sample['density'],(3,))
            #sample['point_maps'] = torch.flip(sample['point_maps'],(3,))
            
        if random.randint(0,1):
            sample['image'] = torch.flip(sample['image'],(2,))
            sample['density'] = torch.flip(sample['density'],(2,))
           #sample['point_maps'] = torch.flip(sample['point_maps'],(2,))
            
        rangle = float(random.randint(0,3)*90)
        sample['image'] = TF.rotate(sample['image'],angle=rangle)
        sample['density'] = TF.rotate(sample['density'],angle=rangle)
        #sample['point_maps'] = TF.rotate(sample['point_maps'],angle=rangle)
        
        sample['image'] = sample['image'].squeeze()
        sample['density'] = sample['density'].squeeze().squeeze()
        
        return sample    

class CustCrop(object):
    """Crop images to match vgg feature sizes."""

    def __call__(self, sample):
              
        if 'density' in sample.keys():
            density = sample['density']
            
            pd = 0
            
            if c.feat_extractor == 'resnet18' and c.downsampling:
                pd = 8
            elif c.feat_extractor == 'alexnet' and c.downsampling:
                density = density[:544,:768]
#            elif c.feat_extractor == 'vgg16_bn':
#                density = density[:544,:768]
            
            # (padding_left,padding_right, padding, padding, padding_top,padding_bottom)
            # density = F.pad(input=density, pad=(0,0,pd,0), mode='constant', value=0)
            sample['density'] = density
            
            if c.debug:
                pass
                #print("image padded by {}".format(pd))
            
            'remove padding and scale instead (CustResize) to prevent artifacts occuring from model'
            # if c.pyramid:
            #     # adding padding so high level features match dmap dims after downsampling (37,38)
            #     image = sample['image']
            #     image = F.pad(input=image, pad=(0,0,pd,0), mode='constant', value=0)
            #     sample['image'] =  image
        
        return sample
    
class CustResize(object):
    """Resize density, according to config scale parameter."""

    def __call__(self, sample):
              
        if 'density' in sample.keys():
            density = sample['density']
            #sz = list(density.size())
            sz = [c.density_map_h,c.density_map_w]
            
            # channels, height, width | alias off by default, bilinear default
            density = density.unsqueeze(0).unsqueeze(0)
            density = TF.resize(density,(sz[0]//c.scale,sz[1]//c.scale))
            density = density.squeeze().squeeze()
            
            if c.pyramid:
                # adding padding so high level features match dmap dims after downsampling (37,38)
                image = sample['image']
                image = TF.resize(image,(sz[0]//c.scale,sz[1]//c.scale))
                sample['image'] =  image
            
            if c.debug:
                print("image scaled down factor by {}".format(c.scale))
            
            sample['density'] = density
        
        return sample

# Need to create lists of test and train indices
# Dataset is highly imbalanced, want test set to mirror train set imbalance 
def train_val_split(dataset,train_percent,oversample=False,
                    balanced = False,annotations_only = False,seed = -1):
    
    ''' 
     Args:
         dataset: pytorch dataset
         train_percent: percentage of dataset to allocate to training set
         balance: whether to create a 50/50 dataset of annotations:empty patches
         annotations_only: create a dataset that only has non-empty patches 
         
    Returns:
        tuple of two lists of shuffled indices - one for train and test
        
    Notes:
        class balance (in terms of proportion of annotations in val & train) is preserved
        iterates over entire dataset (inc. images, so quite slow)
     
     '''
    
    assert not (balanced and oversample)
    assert type(seed) == int
    
    if seed != -1:
        random.seed(seed)
    else:
        ValueError('Must set random seed for evaluation purposes!')
    
    if c.verbose:
        print("Creating indicies...")
    
    l = len(dataset)
    
    v_indices = []; v_weights = []
    t_indices = []; t_weights = []
    
    e_indices = []
    a_indices = []
    
    for i in range(l):
        if len(dataset.get_annotations(i)) != 0:
            a_indices.append(i)
        else:
            e_indices.append(i)
    
    # designed to be used with balanced sampler
    if oversample:
        a_indices = np.tile(np.array(a_indices),int(np.round(len(e_indices)/len(a_indices))))
            
    if c.debug:
        print(len(a_indices))
        print(len(e_indices))
    
    # modify lists to be random
    np.random.shuffle(a_indices)
    np.random.shuffle(e_indices)
    
    if balanced:
        e_indices = e_indices[:len(a_indices)]
    
    split_e = round(train_percent * len(e_indices) / 100)
    split_a = round(train_percent * len(a_indices) / 100)
    
    # add second dimension to indicate empty or not, add weights
    if annotations_only:
        weight_a=1
        weight_e=1
    else:
        # so weight = 1 for both
        weight_a=len(a_indices)
        weight_e=len(e_indices)
    
    a_weights = ((np.ones(len(a_indices))/weight_a).tolist())
    e_weights = ((np.ones(len(e_indices))/weight_e).tolist())
    
    # print("Length Annotated and Eempty Weights")
    # print(len(a_weights))
    # print(len(e_weights))
    # print(a_weights[0])
    # print(e_weights[0])
    
    # only split if not annotations ony
    # TODO: DRY (and ugly)
    if not annotations_only:
        t_indices.extend(e_indices[:split_e])
        t_weights.extend(e_weights[:split_e])
        v_weights.extend(e_weights[split_e:])
        v_indices.extend(e_indices[split_e:])
        
    t_indices.extend(a_indices[:split_a])
    v_indices.extend(a_indices[split_a:])
    
    t_weights.extend(a_weights[:split_a])
    v_weights.extend(a_weights[split_a:])
        
    # shuffle in unison
    t = list(zip(t_indices,t_weights))
    v = list(zip(v_indices, v_weights))
    
    random.shuffle(t)
    t_indices, t_weights = zip(*t)
    
    random.shuffle(v)
    v_indices, v_weights = zip(*v)
 
    t_indices = t_indices[:round(c.data_prop*len(t_indices))]
    t_weights = t_weights[:round(c.data_prop*len(t_weights))]

    v_indices = v_indices[:round(c.data_prop*len(v_indices))]
    v_weights = v_weights[:round(c.data_prop*len(v_weights))]
    
    if c.verbose:
        print("Finished creating indicies")
    
    return t_indices, t_weights, v_indices, v_weights

# demonstrate class and methods:
if demo:

# instantiate class
    cow_dataset = CowObjectsDataset(root_dir=proj_dir,convert_to_points=points_flag,generate_density=density_demo)
    
    if cow_dataset.density:
        key = 'density'
    else:
        key = 'annotations'
    
# test helper 'show annotations' function
    header_list = ['class', 'x', 'y', 'width', 'height']   
    img_name = '930WJ-92NMX_2400_2400.jpg' 
    txt_path = os.path.join(cow_dataset.root_dir,"obj/930WJ-92NMX_2400_2400.txt") 
    annotations = pd.read_csv(txt_path,names=header_list,delim_whitespace=True)
    annotations = np.asarray(annotations)  
    im = Image.open(os.path.join(cow_dataset.root_dir,'obj/', img_name))

    # iterate through the data samples. 
    # we will print the sizes of first 4 samples and show their annotations.
    if False:
        for i in range(len(cow_dataset)):
            sample = cow_dataset[i]
        
            if c.debug:
                # second item will either be annotations or a density map
                print(i, sample['image'].shape, sample[key].shape)
        
            cow_dataset.show_annotations(i,title = 'Sample #{}'.format(i),show = True)
        
            if i == 3:
                break
        
    # since the first images are empty, hit the below until some annotations come up
    if random_flag:
        r_int = random.randint(0, len(cow_dataset))
        cow_dataset.show_annotations(r_int,show = True,title = r_int)
    else:
        cow_dataset.show_annotations(5809,show = True,title = 5809)
    
    transformed_dataset = CowObjectsDataset(root_dir=proj_dir,transform = CustToTensor(),convert_to_points=points_flag,generate_density=density_demo)
        
    # example iterating over the dataset
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        
        if c.debug:
            print(i, sample['image'].size(), sample[key].size())
    
        if i == 3:
            break
    
    # use DataLoader for batching, shuffling, loading images in parallel
    # if density, can use default collate function for batching
    if transformed_dataset.density:
        dataloader = DataLoader(transformed_dataset, batch_size=4,
                                shuffle=True, num_workers=0,collate_fn=cow_dataset.custom_collate_aerial)
    else:
        dataloader = DataLoader(transformed_dataset, batch_size=4,
                                shuffle=True, num_workers=0,collate_fn=cow_dataset.custom_collate_fn)
    
    
    j = 0
    
    for i_batch, sample_batched in enumerate(dataloader):
        r_int = random.randint(0, len(cow_dataset))
        
        if transformed_dataset.density:  
            sb = list()
            sb.append(sample_batched[0])
            sb.append(sample_batched[1])
        else: 
            sb = sample_batched
        
        if c.debug:
            print(i_batch)
            print(sb[0])
            print(sb[1])
        
        # since annotations are sparse, break when some are found:
        if True:
            for i in range(0,len(sb[0])):
                if sb[1][i].sum() != 0:
                    cow_dataset.show_annotations_batch(sb,debug=c.debug)
                    j += 1
                    break
                
        if j == 1: break
    
        # observe 4th batch and stop.
        if i_batch == r_int and random == True:
            cow_dataset.show_annotations_batch(sb,debug=c.debug)
            
        if i_batch == 4:
            cow_dataset.show_annotations_batch(sb,debug=c.debug)