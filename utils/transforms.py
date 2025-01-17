'''
Copyright (c) R. Mineo, 2022-2024. All rights reserved.
This code was developed by R. Mineo in collaboration with PerceiveLab and other contributors.
For usage and licensing requests, please contact the owner.
'''

from cmath import nan
import torch
import copy
from monai.transforms import Randomizable, MapTransform, Transform
from monai.config import KeysCollection
import numpy as np
from skimage.transform import resize
import os
from scipy.io import loadmat

import math

class Convert1Ch(MapTransform):
    def __call__(self, data):
        rgb_weights = [0.2989, 0.5870, 0.1140]
        d = copy.deepcopy(data)
        for key in self.keys:
            if d[key].shape[-1] == 4: 
                rgb_weights = [0.2989, 0.5870, 0.1140, 0]
            d[key]=np.dot(d[key], rgb_weights)
            d[key] = np.expand_dims(d[key], axis = 2)
            d[key]=d[key].astype(float)
        return d
    
class ResizeWithRatio(MapTransform):
    def __init__(self, keys: KeysCollection, image_size):
        super().__init__(keys)
        self.image_size = image_size
        
    def __call__(self, data):
        d = copy.deepcopy(data)
        for key in self.keys:
            if len(d[key].shape) > 3:
                d[key] = d[key][:,:,:3]
            if d[key].shape[0]>d[key].shape[1]:
                h = self.image_size
                hpercent = (h/float(d[key].shape[0]))
                w = int(float(d[key].shape[1])*float(hpercent))
            else:
                w = self.image_size
                wpercent = (w/float(d[key].shape[1]))
                h = int(float(d[key].shape[0])*float(wpercent))
            d[key] = resize(d[key], (h,w,d[key].shape[-1]), anti_aliasing=True)
            d[key]=d[key].astype(float)
        return d
    
class Delete4Ch(MapTransform):
    def __call__(self, data):
        d = copy.deepcopy(data)
        for key in self.keys:
            if d[key].shape[-1] == 4:
                d[key] = d[key][:,:,:3]
        if 'crop' in d.keys(): del d['crop']
        if 'region' in d.keys(): del d['region']
        if 'mask' in d.keys(): del d['mask']
        if 'mask_rect' in d.keys(): del d['mask_rect']
        if 'mask_rect10' in d.keys(): del d['mask_rect10']
        return d
    
class RandPatchedImageWith0Padding(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key, label_key = self.keys
        patch_row = data[img_key].size()[2]
        patch_col = data[img_key].size()[3]
        patched_img = torch.zeros(data[img_key].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,self.start_id+counter,:,:]
                    counter += 1
       
        cropper = copy.deepcopy(data)
        cropper[img_key] = patched_img
        cropper['start_id'] = self.start_id
        return cropper
    
class RandPatchedImage(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key, label_key = self.keys
        patch_row = data[img_key].size()[2]
        patch_col = data[img_key].size()[3]
        patched_img = torch.zeros(data[img_key].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,self.start_id+counter,:,:]
                    counter += 1
                else:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,self.start_id+counter-1,:,:]
       
        cropper = copy.deepcopy(data)
        cropper[img_key] = patched_img
        cropper['start_id'] = self.start_id
        return cropper

class RandPatchedImageLateFusion(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key_T1, img_key_T2, label_key = self.keys
        patch_row = data[img_key_T1].size()[2]
        patch_col = data[img_key_T1].size()[3]
        patched_img_T1 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_img_T2 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key_T1].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter,:,:]
                    counter += 1
                else:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter-1,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter-1,:,:]
       
        cropper = copy.deepcopy(data)
        cropper[img_key_T1] = patched_img_T1
        cropper[img_key_T2] = patched_img_T2
        cropper['start_id'] = self.start_id
        return cropper

class RandPatchedImageAndEarlyFusion(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key_T1, img_key_T2, label_key = self.keys
        patch_row = data[img_key_T1].size()[2]
        patch_col = data[img_key_T1].size()[3]
        patched_img_T1 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_img_T2 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key_T1].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter,:,:]
                    counter += 1
                else:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter-1,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter-1,:,:]
       
        patched_fused_img = torch.cat((patched_img_T1, patched_img_T2), dim = 0)
        cropper = copy.deepcopy(data)
        del cropper[img_key_T1]
        del cropper[img_key_T2]
        cropper['fusedImage'] = patched_fused_img
        cropper['start_id'] = self.start_id
        return cropper

class CenterPatchedImageAndEarlyFusion(MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
      
    def __call__(self, data):
        img_key_T1, img_key_T2, label_key = self.keys
        patch_row = data[img_key_T1].size()[2]
        patch_col = data[img_key_T1].size()[3]
        patched_img_T1 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_img_T2 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key_T1].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else:
            self.start_id = int(num_slices/2)-int(self.num_patches/2)
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter,:,:]
                    counter += 1
                else:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter-1,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter-1,:,:]
       
        patched_fused_img = torch.cat((patched_img_T1, patched_img_T2), dim = 0)
        cropper = copy.deepcopy(data)
        del cropper[img_key_T1]
        del cropper[img_key_T2]
        cropper['fusedImage'] = patched_fused_img
        cropper['start_id'] = self.start_id
        return cropper
        

class RandDepthCrop(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_slices=3):
        super().__init__(keys)
        self.num_slices = num_slices
        self.start_id = 0
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key, label_key = self.keys
        max_value = data[img_key].shape[1] - self.num_slices
        self.randomize(max_value)
        slice_ = data[img_key][0,self.start_id:(self.start_id+self.num_slices),:,:]
        n = slice_.shape[0]
        while n<self.num_slices:
            slice_ = torch.cat([slice_, slice_[-1].unsqueeze(0)],dim = 0)
            n+=1
        cropper = copy.deepcopy(data)
        cropper[img_key] = slice_
        cropper['start_id'] = self.start_id
        return cropper
    
class NewMergedImage(MapTransform):
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
    
    def __call__(self, data):
        d = copy.deepcopy(data)
        img_key_T1, img_key_T2 = self.keys
        img_key_merge = 'merged'
        
        merged_data = d[img_key_T1] - d[img_key_T2]
        d[img_key_merge] = merged_data
        return d
    
class RandPatchedImage3Channels(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key_T1, img_key_T2, label_key = self.keys
        img_key_merge = 'merged'
        patch_row = data[img_key_T1].size()[2]
        patch_col = data[img_key_T1].size()[3]
        patched_img_T1 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_img_T2 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_merge = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key_T1].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter,:,:]
                    patched_merge[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_merge][:,self.start_id+counter,:,:]
                    counter += 1
                else:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter-1,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter-1,:,:]
                    patched_merge[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_merge][:,self.start_id+counter-1,:,:]
                    
        patched_fused_img = torch.cat((patched_img_T1, patched_img_T2, patched_merge), dim = 0)
        cropper = copy.deepcopy(data)
        del cropper[img_key_T1]
        del cropper[img_key_T2]
        del cropper[img_key_merge]
        cropper['fusedImage'] = patched_fused_img
        cropper['start_id'] = self.start_id
        return cropper
    
class NDITKtoNumpy(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
        
    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d[k] = np.asarray(d[k])
        return d

class ElaborateNanD(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
    
    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d[k] = float(d[k])
        return d

class AppendRootDirD(MapTransform):
    def __init__(self, keys: KeysCollection, root_dir):
        super().__init__(keys)
        self.root_dir = root_dir
    
    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d[k] = os.path.join(self.root_dir,d[k])
        return d

class CloneD(MapTransform):
    def __init__(self, keys: KeysCollection, suffix : str):
        super().__init__(keys)
        self.suffix = suffix
    
    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d[(k+self.suffix)] = d[k]
        return d

class ModLabel(Transform):
    def __init__(self, key_dest, key_1, threshold_1, key_2, threshold_2):
        super().__init__()
        self.key_dest = key_dest
        self.key_1 = key_1
        self.threshold_1 = threshold_1
        self.key_2 = key_2
        self.threshold_2 = threshold_2
    
    def __call__(self, data):
        d = copy.deepcopy(data)

        if (self.threshold_1==None or self.threshold_2==None):
            return d

        if (self.threshold_1<0 or self.threshold_1>1 or self.threshold_2<0 or self.threshold_2>1):
            raise RuntimeError("threshold 1 and/or threshold 2 must be in range [0,1].")
        
        if (d[self.key_1] != nan):
            if (d[self.key_1] <= self.threshold_1):
                d[self.key_dest] = 1
            else:
                d[self.key_dest] = 0
        elif (d[self.key_2] != nan):
            if (d[self.key_2] <= self.threshold_2):
                d[self.key_dest] = 1
            else:
                d[self.key_dest] = 0
        else:
            raise RuntimeError("data[key_1] and data[key_2] are both NaN.")
        
        return d
    
class ImageTo2dD(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
    
    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d[k] = d[k][d[k].shape[0]//2,:,:]
        return d
    
class LoadMatD(MapTransform):
    def __init__(self, keys: KeysCollection, key):
        super().__init__(keys)
        self.key = key
    
    def __call__(self, data):
        d = copy.deepcopy(data)
        for k in self.keys:
            d[k] = loadmat(d[k])[self.key].T
            d[k] = d[k].astype('int32')
        return d
    
class NormalizeD(MapTransform):
    def __init__(self, keys: KeysCollection, subtrahend, divisor):
        super().__init__(keys)
        self.subtrahend = np.array(subtrahend)
        self.divisor = np.array(divisor)
    
    def __call__(self, data):
        d = copy.deepcopy(data)
        for i,k in enumerate(self.keys):
            d[k] = (np.array(d[k])-self.subtrahend[i]) / self.divisor[i]
        return d

class ToGridImageD(MapTransform):
    def __init__(self, keys: KeysCollection, num_patches, stride):
        super().__init__(keys)

        self.num_patches = num_patches
        if not math.sqrt(self.num_patches).is_integer():
            raise ValueError("ToGridImage accept only num_patches square number.")
        
        self.num_cols = int(math.sqrt(self.num_patches))
        self.stride = stride
        
    def __call__(self, data):
        d = copy.deepcopy(data)
        for i,k in enumerate(self.keys):
            num_slices = data[k].size()[1]
            if num_slices != ((self.num_patches*self.stride)-(self.stride-1)):
                raise ValueError("ToGridImage accept num_patches*stride slices in input.")
            
            patch_row = data[k].size()[2]
            patch_col = data[k].size()[3]
            grid = torch.zeros(data[k].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)

            counter = 0
            for i in range(self.num_cols):
                for j in range(self.num_cols):
                    grid[:, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[k][:,counter*self.stride,:,:]
                    counter += 1

            d[k] = grid
        return d