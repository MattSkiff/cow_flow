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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from sklearn.preprocessing import normalize # normalise data
from utils import matlab_style_gauss2D
from PIL import Image

import config as c

proj_dir = c.proj_dir
debug = False
random_flag = False
points_flag = True
demo = False
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
        else:        
            annotations = pd.read_csv(txt_path,names=header_list,delim_whitespace=True)
            annotations = annotations.to_numpy()
            
            if self.points:
                # delete height and width columns, which won't be used in density map
                annotations = np.delete(arr = annotations,obj = [3,4],axis = 1) # np 2d: row = axis 0, col = axis 1
                
            if self.density:
                if not self.points:
                    print("Generation of maps requires conversion to points")
                    
                assert self.points  
                
                # running gaussian filter over points as in crowdcount mcnn
                # https://github.com/svishwa/crowdcount-mcnn/blob/master/data_preparation/get_density_map_gaussian.m
                # https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
                
                # map generation
                gauss2dkern = matlab_style_gauss2D(shape = (c.filter_size,c.filter_size),sigma = c.sigma)
        
        sample = {'image': image, 'annotations': annotations}

        if self.transform:
            sample = self.transform(sample)

        return sample

    # Helper function to show a batch
    def show_annotations_batch(self,sample_batched,debug = False,points = False):
        """Show image with annotations for a batch of samples."""
        fig, ax = plt.subplots(4)
        plt.ioff()
        fig.suptitle('Batch from dataloader',y=0.9,fontsize=24)
        fig.set_size_inches(8,6*4)
        fig.set_dpi(100)
        
        images_batch, annotations_batch = \
                sample_batched[0], sample_batched[1]
        batch_size = len(images_batch)
        
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
                if not points:
                    for j in range(0,len(an)):
                        
                        rect = patches.Rectangle(xy = ((an[j,0]*800-0.5*an[j,2]*800),
                                               an[j,1]*600-0.5*an[j,3]*600),
                                               width = an[j,2]*800,height = an[j,3]*600, 
                                               linewidth=1, edgecolor='r', facecolor='none')
                        # Add the patch to the Axes
                        ax[i].add_patch(rect)
                else:
                    num_cols = an.shape[1]
                    if num_cols != 2:
                        ax[i].scatter(an[:,1]*800,an[:,2]*600, s=mk_size,color = 'red') 
                    else:
                        ax[i].scatter(an[:,0]*800,an[:,1]*600, s=mk_size,color = 'red')
                            
     
        plt.show()
            
    # helper function to show annotations + example
    def show_annotations(self,image, annotations,title = "", save = False,show = False,debug = False,points = False):
        """Show image with landmarks"""
        
        an = annotations
        
        fig, ax = plt.subplots()
        # width, height, default DPI = 100
        fig.set_size_inches(8,6)
        fig.set_dpi(100)
        ax.imshow(image)
        # turn off whitespace etc
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        if debug:
            print(an)
        
        # define patch using x1,x2,y1,y2 coords -> map to height and width
        # <object-class> <centre-x> <centre-y> <width> <height>
        if len(an) != 0:
            if not points:
                for i in range(0,len(an)):
                    rect = patches.Rectangle(xy = (an[i,1]*800-0.5*an[i,3]*800,
                                                   an[i,2]*600-0.5*an[i,4]*600),
                                             width = an[i,3]*800,height = an[i,4]*600, 
                                             linewidth=1, edgecolor='r', facecolor='none')
                    # Add the patch to the Axes
                    ax.add_patch(rect)
            else:
                num_cols = an.shape[1]
                if num_cols != 2:
                    ax.scatter(an[:,1]*800,an[:,2]*600, s=mk_size,color = 'red') 
                else:
                    ax.scatter(an[:,0]*800,an[:,1]*600, s=mk_size,color = 'red')
        
        if title != "":
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
    
        images = list()
        boxes = list()
        labels = list()
        
        for b in batch:
            images.append(b['image'])
    
            if b['annotations'].numel() == 0:
                boxes.append(torch.empty(size=(0,4)))
                labels.append(torch.empty(size=(0,1)))
            else:
                bx = b['annotations'][:, 1:]
                if debug:
                    print("bx size:")
                    print(bx.size())
                #boxes.append(bx.resize_((0,torch.numel(bx))))
                boxes.append(bx)
        
                l = b['annotations'][:, 0]
                #labels.append(l.resize_((0,torch.numel(l))))
                labels.append(l)
        
        images = torch.stack(images,dim = 0)
        
        # works, but data is now single tensor instead of tensor of tensors
        
        #boxes = torch.cat(boxes,dim = 1) 
        #labels = torch.cat(labels,dim = 1)
        
        
        if debug:
            print(type(boxes))
            print(type(labels))
            print(type(images))
            print(boxes)
            print(labels)
        #boxes = torch.stack(boxes,dim = 0)
        #labels = torch.stack(boxes,dim = 0 )
    
        return images,boxes,labels

# Define transform to tensor
# Rescale transform not needed as slices are all same size
# RandomCrop not needed (unnecesarry for this task)
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, annotations = sample['image'], sample['annotations']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'annotations': torch.from_numpy(annotations)}

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
    cow_dataset = CowObjectsDataset(root_dir=proj_dir,convert_to_points=points_flag)
    
# test helper 'show annotations' function
    header_list = ['class', 'x', 'y', 'width', 'height']   
    img_name = '930WJ-92NMX_2400_2400.jpg' 
    txt_path = os.path.join(cow_dataset.root_dir,"obj/930WJ-92NMX_2400_2400.txt") 
    annotations = pd.read_csv(txt_path,names=header_list,delim_whitespace=True)
    annotations = np.asarray(annotations)  
    im = Image.open(os.path.join(cow_dataset.root_dir,'obj/', img_name))
    
    cow_dataset.show_annotations(im,annotations,show = True,points=points_flag)

    # iterate through the data samples. 
    # we will print the sizes of first 4 samples and show their annotations.
    for i in range(len(cow_dataset)):
        sample = cow_dataset[i]
    
        if debug:
            print(i, sample['image'].shape, sample['annotations'].shape)
    
        cow_dataset.show_annotations(**sample,title = 'Sample #{}'.format(i),show = True,points=points_flag)
    
        if i == 3:
            break
        
    # since the first images are empty, hit the below until some annotations come up
    if random_flag:
        r_int = random.randint(0, len(cow_dataset))
        cow_dataset.show_annotations(**cow_dataset[r_int],show = True,title = r_int,points=points_flag)
    else:
        cow_dataset.show_annotations(**cow_dataset[5809],show = True,title = 5809,points=points_flag)
    
    transformed_dataset = CowObjectsDataset(root_dir=proj_dir,transform = ToTensor(),convert_to_points=points_flag)
        
    # example iterating over the dataset
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        
        if debug:
            print(i, sample['image'].size(), sample['annotations'].size())
    
        if i == 3:
            break
    
    # use DataLoader for batching, shuffling, loading images in parallel
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=0,collate_fn=cow_dataset.custom_collate_fn)
    
    
    
    for i_batch, sample_batched in enumerate(dataloader):
        r_int = random.randint(0, len(cow_dataset))
        if debug:
            print(i_batch)
            print(sample_batched[0])
            print(sample_batched[1])
        
        # since annotations are sparse, break when some are found:
        if True:
            for i in range(0,len(sample_batched[0])):
                if sample_batched[1][i].size()[0] != 0:
                    cow_dataset.show_annotations_batch(sample_batched,debug=debug,points=points_flag)
                    break
    
        # observe 4th batch and stop.
        if i_batch == r_int and random == True:
            cow_dataset.show_annotations_batch(sample_batched,debug=debug,points=points_flag)
            
        if i_batch == 4:
            cow_dataset.show_annotations_batch(sample_batched,debug=debug,points=points_flag)
    
            break