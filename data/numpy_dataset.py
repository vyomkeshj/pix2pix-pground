import os
import numpy as np
from data.base_dataset import BaseDataset
from data.numpy_loader import load_frames
from PIL import Image
import albumentations as alb


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return r


def create_mask(mask_matrix, id2label_ids):
    """ Takes a mask matrix and a label id, returns a boolean mask with just the areas with the label id """
    # print(mask_matrix)
    final_mask = np.zeros_like(mask_matrix)
    for id in id2label_ids:
        mask = np.ones_like(mask_matrix) * id
        mask = np.equal(mask, mask_matrix)
        final_mask += mask
    return final_mask.astype(np.uint8)


def get_transformed_images_masks(input_image, segementation_channel, thermal_image, transform):
    rgb_channels = input_image[:, :, 0:3]

    trees_mask = create_mask(segementation_channel, [171])
    person_mask = create_mask(segementation_channel, [122, 123, 124, 125])
    railroad_mask = create_mask(segementation_channel, [94])
    sky_mask = create_mask(segementation_channel, [138])

    thermal_stack = np.dstack([thermal_image, thermal_image, thermal_image])
    # print(f"stacked thermal shape: {thermal_stack.shape}")

    transformed = transform(image=rgb_channels,
                            thermal_image=thermal_stack,
                            trees_mask=trees_mask,
                            person_mask=person_mask,
                            railroad_mask=railroad_mask,
                            sky_mask=sky_mask)

    return transformed['image'] / 255., rgb2gray(transformed['thermal_image']) / 255., \
        {
            'person_mask': transformed['person_mask'][..., np.newaxis],
            'trees_mask': transformed['trees_mask'][..., np.newaxis],
            'railroad_mask': transformed['railroad_mask'][..., np.newaxis],
            'sky_mask': transformed['sky_mask'][..., np.newaxis]
        }


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
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image

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
        mask_channel = current_npz_frames['A'][:, :, 3]
        thermal_channel = current_npz_frames['B'][:, :, 0]

        transform = alb.Compose([
            alb.Rotate(limit=30, interpolation=1, border_mode=4, value=None, mask_value=None, rotate_method='largest_box', crop_border=False, always_apply=True, p=0.5),
            alb.RandomCrop(width=512, height=512),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.3),
            alb.RGBShift(r_shift_limit=60, g_shift_limit=60, b_shift_limit=60, always_apply=False, p=0.7),
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

        # print(f"transformed_image shape: {transformed_image.shape}")

        return {'rgb_channels': transformed_image,
                'thermal_channel': transformed_thermal,
                'mask_dict': mask_dict}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.input_files)
