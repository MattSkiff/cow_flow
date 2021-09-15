# we are following the YOLO format for data formatting and bounding box annotations
# see here: https://github.com/AlexeyAB/Yolo_mark/issues/60 for a description

# edited from:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Author: Sasank Chilamkurthy

# edit tutorial to work for object annotations and YOLO format
import os 
import torch 
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# https://docs.opencv.org/4.5.2/d4/d13/tutorial_py_filtering.html
import scipy

from torch.utils.data import Dataset, DataLoader

# transforms = todo
#from torchvision import transforms
#from skimage import transform
#from sklearn.preprocessing import normalize # normalise data

from torchvision import utils
from skimage import io
from PIL import Image

import config as c

proj_dir = c.proj_dir
random_flag = False
points_flag = False
demo = False
density_demo = False
mk_size = 25

class CowObjectsDataset(Dataset):
    """Cow Objects dataset."""

    def __init__(self, root_dir,transform=None,convert_to_points=False,generate_density=False):
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
            
        self.train_im_paths = train_im_paths
        
        
    def __len__(self):
        with open(os.path.join(self.root_dir,"train.txt")) as f:
                  train_ims = f.readlines()
        
        
        return len(train_ims)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        labels = []

        img_path = os.path.join(self.root_dir,
                                self.train_im_paths[idx])
        
        txt_path = img_path[:-3]+'txt'

        # yolo object annotation columns
        # <object-class> <x> <y> <width> <height>
        image = io.imread(img_path)
        
        # check if annotations file is empty
        header_list = ['class', 'x', 'y', 'width', 'height']
        
        if os.stat(txt_path).st_size == 0:
            annotations = np.array([])
            density_map = np.zeros((c.img_size[1], c.img_size[0]), dtype=np.uint8)
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
                # gauss2dkern = matlab_style_gauss2D(shape = (c.filter_size,c.filter_size),sigma = c.sigma)
                density_map = np.zeros((c.img_size[1], c.img_size[0]), dtype=np.float32)
               
                # add points onto basemap
                for point in annotations:
                    # error introduced here as float position annotation centre converted to int
                    
                    base_map = np.zeros((c.img_size[1], c.img_size[0]), dtype=np.float32)
                    
                    if c.debug:
                        print(point)
                        
                    # subtract 1 to account for 0 indexing
                    base_map[int(round(point[2]*c.img_size[1])-1),int(round(point[1]*c.img_size[0])-1)] += 1
                    density_map += scipy.ndimage.filters.gaussian_filter(base_map, sigma = c.sigma, mode='constant')
                    
                    labels.append(point[0])
                    
                    if c.debug:
                        print(base_map.sum())
                        print(density_map.sum())
                
        labels = np.array(labels) # list into default collate function produces empty tensors
        
        if not self.density:
            sample = {'image': image, 'annotations': annotations}
            
        if self.density:  
            sample = {'image': image, 'density': density_map, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

    # Helper function to show a batch
    def show_annotations_batch(self,sample_batched,debug = False):
        
        """Show image with annotations for a batch of samples."""
        
        batch_size = len(sample_batched[0])
        
        if self.density: 
            ncol = 2 
        else:
            ncol = 1
        
        fig, ax = plt.subplots(batch_size,ncol)
        plt.ioff()
        fig.suptitle('Batch from dataloader',y=0.9,fontsize=24)
        fig.set_size_inches(8*ncol,6*batch_size)
        fig.set_dpi(100)
      
        images_batch = sample_batched[0]
      
        if self.density:
            
            density_batch = sample_batched[1]
            
            for i in range(batch_size):
                density = density_batch[i]
                image = images_batch[i]
                image = image.permute((1, 2, 0))
                ax[i,0].axis('off')
                ax[i,1].axis('off')
                ax[i,0].imshow(density, cmap='hot', interpolation='nearest')
                ax[i,1].imshow(image)
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
        
        if self.density:
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(sample['density'], cmap='hot', interpolation='nearest')
                ax[1].imshow(sample['image'], cmap='hot', interpolation='nearest')
        else:
            """Show image with landmarks"""
            image = sample['image']
            an = sample['annotations']
            
            fig, ax = plt.subplots()
            # width, height, default DPI = 100
            fig.set_size_inches(8,6)
            fig.set_dpi(100)
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
            
    # because batches have varying numbers of bounding boxes, need to define custom collate func
    # https://discuss.pytorch.org/t/dataloader-gives-stack-expects-each-tensor-to-be-equal-size-due-to-different-image-has-different-objects-number/91941/5
    
    # method to stack tensors of different sizes:
    # https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py
    
    def custom_collate_fn(self,batch,debug = False):
        # only needed to collate annotations

        images = list()
        boxes = list()
        labels = list()
        
        for b in batch:
            images.append(b['image'])
    
            if b['annotations'].numel() == 0:
                boxes.append(torch.empty(size=(0,4)))

            else:
                bx = b['annotations'][:, 1:]
                if debug:
                    print("bx size:")
                    print(bx.size())
                    
                boxes.append(bx)
        
        labels.append(b['annotations'][:, 0]) # append labels regardless
        
        images = torch.stack(images,dim = 0)

        if debug:
            print(type(boxes))
            print(type(labels))
            print(type(images))
            print(boxes)
            print(labels)
    
        return images,boxes,labels

    def custom_collate_density(self,batch,debug = False):
        # only needed to collate annotations

        images = list()
        density = list()
        labels = list()
        
        for b in batch:
            images.append(b['image'])
            density.append(b['density'])
            labels.append(b['labels'])

        images = torch.stack(images,dim = 0)
        density = torch.stack(density,dim = 0)
        labels = np.array(labels, dtype=object)
        
        if debug:
            print(type(density))
            print(type(labels))
            print(type(images))
            print(density)
    
        return images,density,labels

# Define transform to tensor
# Rescale transform not needed as slices are all same size
# RandomCrop not needed (unnecesarry for this task)
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
             
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        
        sample['image'] =  torch.from_numpy(image)
        
        if 'density' in sample.keys():
            sample['density'] = torch.from_numpy(sample['density'])
        else:
            sample['annotations'] = torch.from_numpy(sample['annotations'])
            sample['labels'] = torch.from_numpy(sample['labels'])
        
        return sample

# Need to create lists of test and train indices
# Dataset is highly imbalanced, want test set to mirror train set imbalance
# TODO: implement 'balanced' argument
# TODO: get split function to only iterate over txt files, not images 
def train_valid_split(dataset,train_percent,balanced = False,annotations_only = False):
    ''' 
     Args:
         dataset: pytorch dataset
         train_percent: percentage of dataset to allocate to training set
         balance: whether to create a 50/50 dataset of annotations:empty patches
         annotations_only: create a dataset that only has non-empty patches 
         
    Returns:
        tuple of two lists of shuffled indices - one for train and test
        
    Notes:
        class balance (in terms of proportion of annotations in valid & train) is preserved
        iterates over entire dataset (inc. images, so quite slow)
     
     '''
    
    l = len(dataset)
    
    valid_indices = []
    train_indices = []
    

    empty_indices = []
    annotation_indices = []
    
    for i in range(l):
        if len(dataset[i]['labels']) != 0:
            annotation_indices.append(i)
        else:
            empty_indices.append(i)
            
    if c.debug:
        print(len(annotation_indices))
        print(len(empty_indices))
    # modify lists to be random
    np.random.shuffle(annotation_indices)
    np.random.shuffle(empty_indices)
    
    split_e = round(train_percent * len(empty_indices) / 100)
    split_a = round(train_percent * len(annotation_indices) / 100)
    
    
    if not annotations_only:
        train_indices.extend(empty_indices[:split_e])
        
    train_indices.extend(annotation_indices[:split_a])
    
    if not annotations_only:
        valid_indices.extend(empty_indices[split_e:])
        
    valid_indices.extend(annotation_indices[split_a:])
    
    np.random.shuffle(valid_indices)
    np.random.shuffle(train_indices)
            
    return train_indices, valid_indices

# TODO
# class Normalize(object):
#     """Normalise ndarrays in sample"""
    
#     def __call__(self, sample):
#         image, annotations = sample['image'], sample['annotations']
        
#         image = image.transpose((2, 0, 1))
#         return {'image': torch.from_numpy(image),
#                 'annotations': torch.from_numpy(annotations)

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
    
    transformed_dataset = CowObjectsDataset(root_dir=proj_dir,transform = ToTensor(),convert_to_points=points_flag,generate_density=density_demo)
        
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
                                shuffle=True, num_workers=0,collate_fn=cow_dataset.custom_collate_density)
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