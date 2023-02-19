import sys
import os
import json
import torch
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler
from monai.data import CacheDataset, DataLoader, DistributedSampler
from .transforms import (
    AppendRootDirD,
    ElaborateNanD,
    CloneD)
from monai.transforms import (
    Compose,
    AddChannelD,
    LoadImageD,
    ScaleIntensityD,
    NormalizeIntensityD,
    RandFlipD,
    RandRotate90D,
    ResizeD,
    ResizeWithPadOrCropD,
    ToTensorD)
import argparse

class FFRDataset3D(CacheDataset):
    def __init__(self, split_path, section, num_fold, transforms, inner_loop = 0, cache_num = sys.maxsize, cache_rate=1.0, num_workers=1):    
        self.section = section
        self.inner_loop = inner_loop
        self.num_fold = num_fold
        
        data = self._generate_data_list(split_path)

        super().__init__(data, transforms, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)
        
     
    #split data in train, val and test sets in a reproducible way
    def _generate_data_list(self, split_path):
        with open(split_path) as fp:
           path=json.load(fp)
        data = list()
        
        if self.section == 'test':
            data = path[f'fold{self.num_fold}']['test']
        elif self.section == 'training':
            data = path[f'fold{self.num_fold}'][f'inner{self.inner_loop}']['train']
        elif self.section == 'validation':
            data = path[f'fold{self.num_fold}'][f'inner{self.inner_loop}']['val']
        else: 
            raise ValueError(
                    f"Unsupported section: {self.section}, "
                    "available options are ['training', 'validation', 'test']."
                )
        
        #if platform.system() != 'Windows':
        #    for sample in data:
        #        for key in sample.keys():
        #            sample[key] = sample[key].replace('\\', '/')
        return data     
    
    def get_label_proportions(self):
        c = [None]*2
        label_props = [None]*2
        for i in range(2):
            c[i] = len([el['label'] for el in self.data if el['label'] == i])
        for i in range(len(c)):
            label_props[i] = max(c)/c[i]
        return label_props


def get_loader(args):
    if not os.path.isdir(args.root_dir):
        raise ValueError("Root directory root_dir must be a directory.")

    imgs_keys = ['image']
    
    imgs_keys_3d = [x+"_3d" for x in imgs_keys]
    
    basics_1 = [
        ElaborateNanD(['FFR', 'iFR']),
        ToTensorD(['FFR', 'iFR'],dtype=torch.float),
        ToTensorD(['label'],dtype=torch.long),
        AppendRootDirD(imgs_keys, args.root_dir),
        CloneD(imgs_keys, "_3d"),
        LoadImageD(imgs_keys_3d),
        AddChannelD(imgs_keys_3d),
        ResizeD(imgs_keys_3d, spatial_size = args.resize),
        ScaleIntensityD(imgs_keys_3d),
        NormalizeIntensityD(imgs_keys_3d, subtrahend=args.mean, divisor=args.std, channel_wise=True),
        ResizeWithPadOrCropD(imgs_keys_3d, spatial_size=args.pad, mode="constant", method="symmetric"),
        ToTensorD(imgs_keys_3d)
        ]
    
    train_transforms = []
    train_transforms = [
        *basics_1
        ]

    val_transforms = []
    val_transforms = [
        *basics_1
        ]

    train_transforms_img_keys = [*imgs_keys_3d]
    train_transforms = [
        *train_transforms,
        RandFlipD(train_transforms_img_keys, prob = 0.5, spatial_axis=-2),
        RandFlipD(train_transforms_img_keys, prob = 0.5, spatial_axis=-1),
        RandRotate90D(train_transforms_img_keys, prob = 0.5, spatial_axes=(-2,-1))
        ]
    
    train_transforms = Compose(train_transforms)
    val_transforms = Compose(val_transforms)
    
    dataset = {}
    dataset["train"] = FFRDataset3D(split_path = args.split_path, section = 'training', num_fold = args.num_fold, transforms = train_transforms, inner_loop = args.inner_loop, cache_rate=args.cache_rate)
    dataset["validation"] = FFRDataset3D(split_path = args.split_path, section = 'validation', num_fold = args.num_fold, transforms = val_transforms, inner_loop = args.inner_loop, cache_rate=args.cache_rate)
    dataset["test"] = FFRDataset3D(split_path = args.split_path, section = 'test', num_fold = args.num_fold, transforms = val_transforms, inner_loop = args.inner_loop, cache_rate=args.cache_rate)
    
    samplers = {}
    if args.distributed:
        samplers["train"] = DistributedSampler(dataset["train"], shuffle=True)
        samplers["validation"] = DistributedSampler(dataset["validation"], shuffle=False)
        samplers["test"] = DistributedSampler(dataset["test"], shuffle=False)
    else:
        samplers["train"] = RandomSampler(dataset["train"])
        samplers["validation"] = SequentialSampler(dataset["validation"])
        samplers["test"] = SequentialSampler(dataset["test"])

    batch_sampler = {}
    batch_sampler['train'] = BatchSampler(samplers["train"], args.batch_size, drop_last=True)
    batch_sampler['validation'] = BatchSampler(samplers["validation"], args.batch_size, drop_last=False)
    batch_sampler['test'] = BatchSampler(samplers["test"], args.batch_size, drop_last=False)

    loaders = {}
    loaders["train"] = DataLoader(dataset["train"], batch_sampler = batch_sampler['train'], num_workers=2, pin_memory=True, persistent_workers=True)
    loaders["validation"] = DataLoader(dataset["validation"], batch_sampler = batch_sampler['validation'], num_workers=2, pin_memory=True, persistent_workers=True)
    loaders["test"] = DataLoader(dataset["test"], batch_sampler = batch_sampler['test'], num_workers=2, pin_memory=True, persistent_workers=True)
    
    loss_weights = torch.Tensor(dataset["train"].get_label_proportions())

    return loaders, samplers, loss_weights

import matplotlib.pyplot as plt
if __name__ == '__main__':
    args={}
    args['resize'] = (-1,256,256)
    args['pad'] = (60,256,256)
    args['mean'] = 0.0
    args['std'] = 1.0
    args['root_dir'] = os.path.join('..\\data\\MY_DATASET')
    args['split_path'] = os.path.join('..\\data\\MY_DATASET.json')
    args['num_fold'] = 0
    args['inner_loop'] = 0
    args['distributed'] = False
    args['batch_size'] = 8
    args = argparse.Namespace(**args)

    # Dataset e Loader
    loaders, samplers, loss_weights = get_loader(args)

    # Get samples
    tmp = next(iter(loaders["train"]))

    print(tmp['image'][0])
    print(tmp['FFR'][0])
    print(tmp['iFR'][0])
    print(tmp['label'][0])
    
    print("Dataset3D")
    for i in range(tmp['image_3d'][0].numpy().shape[1]):
        plt.imshow(tmp['image_3d'][0].numpy()[0,i,:,:])
        plt.show()
    for i in range(tmp['image_3d'][0].numpy().shape[1]):
        plt.imshow(tmp['image_3d'][0].numpy()[0,i,:,:])
        plt.show()