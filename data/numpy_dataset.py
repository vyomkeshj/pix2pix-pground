import albumentations as alb
import numpy as np
import os
import torch
from data.base_dataset import BaseDataset
from data.utils import load_frames, get_transform


def get_transformed_images_masks(input_image, thermal_image, transform):
    rgb_channels = input_image[:, :, 0:3]

    thermal_stack = np.dstack([thermal_image, thermal_image, thermal_image])

    transformed = transform(image=rgb_channels, thermal_image=thermal_stack)

    rgb_normalizer = get_transform(False)
    transformed_image = torch.permute(rgb_normalizer(transformed['image']), (1, 2, 0))
    gray_normalizer = get_transform(True)
    transformed_thermal = torch.permute(gray_normalizer(transformed['thermal_image']), (1, 2, 0))

    return transformed_image, transformed_thermal[:, :, 0]


class NumpyDataset(BaseDataset):
    """A dataset class to load data from a folder with .npz files [RGB+M_seg]

    It assumes that the directory '/path/to/data/train' contains npz files.
    The test data should be stored in '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.data_path = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.input_files = sorted(load_frames(self.data_path, opt.max_dataset_size))  # get image paths

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        """Return a rgb frame, corresponding masks and thermal frame

        Parameters:
            index - - a random integer for data indexing

        """
        # read a image given a random integer index
        npz_path = self.input_files[index]
        current_npz_frames = np.load(npz_path)

        rgb_channels = current_npz_frames['A'][:, :, 0:3]
        thermal_channel = current_npz_frames['B'] # [:, :, 0]

        transform = alb.Compose([
            alb.Rotate(limit=20, interpolation=1, border_mode=4, value=None, mask_value=None,
                       rotate_method='largest_box', crop_border=False, always_apply=True, p=0.5),
            alb.RandomCrop(width=512, height=512),
            alb.HorizontalFlip(p=0.5),
            alb.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.7),
            alb.CLAHE(clip_limit=2.0, tile_grid_size=(3, 3), always_apply=True),
        ], additional_targets={
            'image': 'image',
            'thermal_image': 'mask',
        })
        transformed_image, transformed_thermal = get_transformed_images_masks(rgb_channels, thermal_channel, transform)

        return {'rgb_channels': transformed_image, 'thermal_channel': transformed_thermal}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.input_files)
