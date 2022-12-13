import os
import numpy as np
from data.base_dataset import BaseDataset #, get_params, get_transform
from data.numpy_loader import load_frames
from PIL import Image
import albumentations as alb
import torch
from data.numpy_dataset import create_mask, rgb2gray

def get_transformed_images_masks(input_image, segementation_channel, thermal_image, transform):
    rgb_channels = input_image[:, :, 0:3]

    trees_mask = create_mask(segementation_channel, [97])
    person_mask = create_mask(segementation_channel, [102])
    railroad_mask = create_mask(segementation_channel, [104, 106])
    sky_mask = create_mask(segementation_channel, [0])

    thermal_stack = np.dstack([thermal_image, thermal_image, thermal_image])
    # print(f"stacked thermal shape: {thermal_stack.shape}")

    transformed = transform(image = rgb_channels,
                            thermal_image = thermal_stack,
                            trees_mask = trees_mask,
                            person_mask = person_mask,
                            railroad_mask = railroad_mask,
                            sky_mask = sky_mask )

    return transformed['image']/255., rgb2gray(transformed['thermal_image'])/255., \
{
'person_mask': transformed['person_mask'][..., np.newaxis],
'trees_mask': transformed['trees_mask'][..., np.newaxis],
'railroad_mask': transformed['railroad_mask'][..., np.newaxis],
'sky_mask': transformed['sky_mask'][..., np.newaxis]
}

class ValidationDataset:

    def __init__(self, path):
        self.data_path = path # os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.input_files = sorted(load_frames(self.data_path, float("inf")))  # get image paths

        self.input_nc = 7 # self.opt.input_nc
        self.output_nc = 3 # self.opt.output_nc


    def __getitem__(self, index):
        current_npz_frames = np.load(self.input_files[index])

        rgb_channels = current_npz_frames['A'][:, :, 0:3]
        mask_channel = current_npz_frames['A'][:, :, 3]
        thermal_channel = current_npz_frames['B'][:, :, 0]
        transform = alb.Compose([
            alb.RandomCrop(width=512, height=512),
            # alb.RGBShift (r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, always_apply=True),
            alb.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), always_apply=True),
        ], additional_targets={
            'image': 'image',
            'thermal_image': 'image',
            'person_mask': 'mask',
            'trees_mask': 'mask',
            'railroad_mask': 'mask',
            'sky_mask': 'mask'})

        transformed_image, transformed_thermal, mask_dict = get_transformed_images_masks(rgb_channels,
                                                                                         mask_channel,
                                                                                         thermal_channel,
                                                                                         transform)

        mask_dict = dict(map(lambda item: (item[0], torch.tensor(item[1][np.newaxis, ...])), mask_dict.items()))
        return {'rgb_channels': torch.tensor(transformed_image[np.newaxis, ...]),
                'thermal_channel': torch.tensor(transformed_thermal[np.newaxis, ...]),
                'mask_dict': mask_dict}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.input_files)
