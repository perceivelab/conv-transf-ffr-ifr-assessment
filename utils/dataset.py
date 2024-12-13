import sys
import os
import json
import platform
import torch
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler # , DataLoader, DistributedSampler
from monai.data import CacheDataset, DataLoader, DistributedSampler
from .transforms import (
    AppendRootDirD,
    ElaborateNanD,
    CloneD,
    ModLabel,
    ImageTo2dD,
    ToGridImageD)
from monai.transforms import (
    Compose,
    AddChannelD,
    RepeatChannelD,
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
        self.num_classes = 2
        
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
        c = [None]*self.num_classes
        label_props = [None]*self.num_classes
        for i in range(self.num_classes):
            c[i] = len([None for data in self.data if data['label'] == i])
        for i in range(len(c)):
            label_props[i] = max(c)/c[i]
        return label_props


def get_loader(args, only_test=False):
    if not os.path.isdir(args.root_dir):
        raise ValueError("Root directory root_dir must be a directory.")

    if (args.datasetGrid) and (args.dataset2d):
        raise ValueError("DatasetGrid can only used with dataset 3D.")

    imgs_keys = ['image']
    if args.doubleViewInput:
        imgs_keys.append('image2')
    
    if args.dataset3d:
        imgs_keys_3d = [x+"_3d" for x in imgs_keys]
    if args.dataset2d:
        imgs_keys_2d = [x+"_2d" for x in imgs_keys]
    
    basics_1 = []
    basics_1 = [
        ElaborateNanD(['FFR', 'iFR']),
        ModLabel('label', 'FFR', args.ffr_threshold, 'iFR', args.ifr_threshold),
        ToTensorD(['FFR', 'iFR'],dtype=torch.float),
        ToTensorD(['label'],dtype=torch.long),
        AppendRootDirD(imgs_keys, args.root_dir),
        ]
    
    if args.enable_clinicalData:
        basics_1 = [
            *basics_1,
            ToTensorD(['age_array', 'sex', 'ckd', 'ef_array', 'stemi', 'nstemi', 'ua', 'stable_angima', 'positive_stress_test'],dtype=torch.float),
        ]
    
    if args.dataset3d:
        basics_1 = [
            *basics_1,
            CloneD(imgs_keys, "_3d"),
            LoadImageD(imgs_keys_3d),
            AddChannelD(imgs_keys_3d),
            ResizeD(imgs_keys_3d, spatial_size = args.resize),
            ScaleIntensityD(imgs_keys_3d),
            ]
        
        if (args.inputChannel == 1):
            basics_1 = [
                *basics_1,
                NormalizeIntensityD(imgs_keys_3d, subtrahend=[torch.Tensor(args.mean).mean(dim=0).item()], divisor=[torch.Tensor(args.std).mean(dim=0).item()], channel_wise=True),
                ]
        else: # args.inputChannel == 3
            basics_1 = [
                *basics_1,
                RepeatChannelD(imgs_keys_3d, repeats=3),
                NormalizeIntensityD(imgs_keys_3d, subtrahend=args.mean, divisor=args.std, channel_wise=True),
                ]
    if args.dataset2d:
        basics_1 = [
            *basics_1,
            CloneD(imgs_keys, "_2d"),
            LoadImageD(imgs_keys_2d),
            ImageTo2dD(imgs_keys_2d),
            AddChannelD(imgs_keys_2d),
            ResizeD(imgs_keys_2d, spatial_size = args.resize[1:3]), # size_mode="longest"
            ScaleIntensityD(imgs_keys_2d),
            ]

        if (args.inputChannel == 1):
            basics_1 = [
                *basics_1,
                NormalizeIntensityD(imgs_keys_2d, subtrahend=[torch.Tensor(args.mean).mean(dim=0).item()], divisor=[torch.Tensor(args.std).mean(dim=0).item()], channel_wise=True),
                ]
        else: # args.inputChannel == 3
            basics_1 = [
                *basics_1,
                RepeatChannelD(imgs_keys_2d, repeats=3),
                NormalizeIntensityD(imgs_keys_2d, subtrahend=args.mean, divisor=args.std, channel_wise=True),
                ]
    
    if args.dataset3d:        
        basics_1 = [
            *basics_1,
            ResizeWithPadOrCropD(imgs_keys_3d, spatial_size=args.pad, mode="constant", method="symmetric"),# attenzione se il modello impara a classificare in base al pad
            ToTensorD(imgs_keys_3d)
            ]
    if args.dataset2d:
        basics_1 = [
            *basics_1,
            ToTensorD(imgs_keys_2d)
            ]
    
    train_transforms = []
    train_transforms = [
        *basics_1
        ]

    val_transforms = []
    val_transforms = [
        *basics_1
        ]

    if args.dataset3d or args.dataset2d:
        train_transforms_img_keys = []
        if args.dataset3d:
            train_transforms_img_keys = [*train_transforms_img_keys, *imgs_keys_3d]
        if args.dataset2d:
            train_transforms_img_keys = [*train_transforms_img_keys, *imgs_keys_2d]
        train_transforms = [
            *train_transforms,
            RandFlipD(train_transforms_img_keys, prob = 0.5, spatial_axis=-2),
            RandFlipD(train_transforms_img_keys, prob = 0.5, spatial_axis=-1),
            RandRotate90D(train_transforms_img_keys, prob = 0.5, spatial_axes=(-2,-1))
            ]

    if args.dataset3d:
        if args.datasetGrid:
            train_transforms = [
                *train_transforms,
                ToGridImageD(imgs_keys_3d, num_patches=args.datasetGridPatches, stride=args.datasetGridStride)
                ]
            val_transforms = [
                *val_transforms,
                ToGridImageD(imgs_keys_3d, num_patches=args.datasetGridPatches, stride=args.datasetGridStride)
                ]
    
    train_transforms = Compose(train_transforms)
    val_transforms = Compose(val_transforms)
    
    dataset = {}
    if not only_test:
        dataset["train"] = FFRDataset3D(split_path = args.split_path, section = 'training', num_fold = args.num_fold, transforms = train_transforms, inner_loop = args.inner_loop, cache_rate=args.cache_rate)
        dataset["validation"] = FFRDataset3D(split_path = args.split_path, section = 'validation', num_fold = args.num_fold, transforms = val_transforms, inner_loop = args.inner_loop, cache_rate=args.cache_rate)
    dataset["test"] = FFRDataset3D(split_path = args.split_path, section = 'test', num_fold = args.num_fold, transforms = val_transforms, inner_loop = args.inner_loop, cache_rate=args.cache_rate)
    
    samplers = {}
    if args.distributed:
        if not only_test:
            samplers["train"] = DistributedSampler(dataset["train"], shuffle=True)
            samplers["validation"] = DistributedSampler(dataset["validation"], shuffle=False)
        samplers["test"] = DistributedSampler(dataset["test"], shuffle=False)
    else:
        if not only_test:
            samplers["train"] = RandomSampler(dataset["train"])
            samplers["validation"] = SequentialSampler(dataset["validation"])
        samplers["test"] = SequentialSampler(dataset["test"])

    batch_sampler = {}
    if not only_test:
        batch_sampler['train'] = BatchSampler(samplers["train"], args.batch_size, drop_last=True)
        batch_sampler['validation'] = BatchSampler(samplers["validation"], args.batch_size, drop_last=False)
    batch_sampler['test'] = BatchSampler(samplers["test"], args.batch_size, drop_last=False)

    loaders = {}
    if not only_test:
        loaders["train"] = DataLoader(dataset["train"], batch_sampler = batch_sampler['train'], num_workers=2, pin_memory=True, persistent_workers=True)
        loaders["validation"] = DataLoader(dataset["validation"], batch_sampler = batch_sampler['validation'], num_workers=2, pin_memory=True, persistent_workers=True)
    loaders["test"] = DataLoader(dataset["test"], batch_sampler = batch_sampler['test'], num_workers=2, pin_memory=True, persistent_workers=True)
    
    loss_weights = torch.Tensor(dataset["train"].get_label_proportions()) if not only_test else None

    return loaders, samplers, loss_weights

import matplotlib.pyplot as plt
if __name__ == '__main__':
    args={}
    args['resize'] = (-1,256,256)
    args['pad'] = (60,256,256)
    args['mean'] = (0.43216, 0.394666, 0.37645)
    args['std'] = (0.22803, 0.22145, 0.216989)
    args['root_dir'] = os.path.join('..\\data')
    args['split_path'] = os.path.join('..\\data\\BNCV5F.json')
    args['num_fold'] = 0
    args['inner_loop'] = 0
    args['distributed'] = False
    args['batch_size'] = 2
    args['ffr_threshold'] = None
    args['ifr_threshold'] = None
    args['doubleViewInput'] = True
    args['enable_clinicalData'] = True
    args['inputChannel'] = 3
    args['dataset3d'] = True
    args['dataset2d'] = True
    args['datasetGrid'] = False
    args['datasetGridPatches'] = 16
    args['datasetGridStride'] = 4
    args['cache_rate'] = 1.0
    args = argparse.Namespace(**args)

    # Dataset e Loader
    loaders, samplers, loss_weights = get_loader(args)

    # Get samples
    tmp = next(iter(loaders["train"]))

    print(tmp['image'][0])
    print(tmp['FFR'][0])
    print(tmp['iFR'][0])
    print(tmp['label'][0])
    print(tmp['age_array'][0])
    if args.dataset3d:
        print("Dataset3D")
        for i in range(tmp['image_3d'][0].numpy().shape[1]):
            plt.imshow(tmp['image_3d'][0].numpy()[0,i,:,:])
            plt.show()
        for i in range(tmp['image_3d'][0].numpy().shape[1]):
            plt.imshow(tmp['image_3d'][0].numpy()[0,i,:,:])
            plt.show()
    if args.dataset3d and args.datasetGrid:
        print("Dataset3DGrid")
        plt.imshow(tmp['image_3d'][0].numpy()[0,:,:])
        plt.show()
        plt.imshow(tmp['image_3d'][0].numpy()[0,:,:])
        plt.show()
    if args.dataset2d:
        print("Dataset2D")
        plt.imshow(tmp['image_2d'][0].numpy()[0,:,:])
        plt.show()
        plt.imshow(tmp['image2_2d'][0].numpy()[0,:,:])
        plt.show()