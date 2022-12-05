import os
import numpy as np
from data.base_dataset import BaseDataset #, get_params, get_transform
from data.numpy_dataset import get_transformed_images_masks
from data.numpy_loader import load_frames
from PIL import Image
import albumentations as alb
import torch

class TestDataset:

    def __init__(self, path):
        self.data_path = path # os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.input_files = sorted(load_frames(self.data_path, float("inf")))  # get image paths

        self.input_nc = 7 # self.opt.input_nc
        self.output_nc = 3 # self.opt.output_nc
        self.transform = alb.Compose([
            alb.RandomCrop(width=512, height=512),
        ], additional_targets={
            'image': 'image',
            'thermal_image': 'image',
            'person_mask': 'mask',
            'trees_mask': 'mask',
            'railroad_mask': 'mask',
            'sky_mask': 'mask'})

    def __getitem__(self, index):
        current_npz_frames = np.load(self.input_files[index])

        rgb_channels = current_npz_frames['A'][:, :, 0:3]
        mask_channel = current_npz_frames['A'][:, :, 3]
        thermal_channel = current_npz_frames['B'][:, :, 0]

        transformed_image, transformed_thermal, mask_dict = get_transformed_images_masks(rgb_channels,
                                                                                         mask_channel,
                                                                                         thermal_channel,
                                                                                         self.transform)
        
        mask_dict = dict(map(lambda item: (item[0], torch.tensor(item[1][np.newaxis, ...])), mask_dict.items()))
        return {'rgb_channels': torch.tensor(transformed_image[np.newaxis, ...]),
                'thermal_channel': torch.tensor(transformed_thermal[np.newaxis, ...]),
                'mask_dict': mask_dict}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.input_files)
