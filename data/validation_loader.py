import albumentations as alb
import numpy as np
import os
import torch
from PIL import Image

from data.base_dataset import BaseDataset  # , get_params, get_transform
# from data.numpy_dataset import create_mask, rgb2gray
from data.numpy_loader import load_frames

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

class ValidationDataset:

    def __init__(self, path):
        self.data_path = path  # os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.input_files = sorted(load_frames(self.data_path, float("inf")))  # get image paths

        self.input_nc = 1  # self.opt.input_nc
        self.output_nc = 1  # self.opt.output_nc


    def __getitem__(self, index):
        current_npz_frames = np.load(self.input_files[index])
        image_channel = current_npz_frames['A'][:, :, 0:3]
        thermal_channel = current_npz_frames['B'][:, :, 0]

        transform = alb.Compose([
            alb.RandomCrop(width=512, height=512),
            alb.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True),
        ], additional_targets={
            'image': 'image',
            'thermal_image': 'image',
        })

        transformed = transform(image=image_channel,
                                thermal_image=thermal_channel)

        # print(f"transformed_image shape: {transformed_image.shape}")

        return {'seg_channel': rgb2gray(transformed['image'])/255.,
                'thermal_channel': transformed['thermal_image']/255.}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.input_files)
