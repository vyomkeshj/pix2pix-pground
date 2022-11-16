import os
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.numpy_loader import load_frames
from PIL import Image


def create_mask(mask_matrix, id2label_id):
    """ Takes a mask matrix and a label id, returns a boolean mask with just the areas with the label id """
    mask = np.ones_like(mask_matrix) * id2label_id
    mask = 255. * np.equal(mask, mask_matrix)
    return mask[...].astype(np.uint8)


def get_mask_dictionary(segementation_channel, mask_transform):
    trees_mask = create_mask(segementation_channel, 100)
    person_mask = create_mask(segementation_channel, 127)

    trees_mask = Image.fromarray(trees_mask).convert('RGB')
    person_mask = Image.fromarray(person_mask).convert('RGB')

    trees_mask = np.array(mask_transform(trees_mask))[0, :, :][np.newaxis, ...]
    person_mask = np.array(mask_transform(person_mask))[0, :, :][np.newaxis, ...]

    return {'person_mask': person_mask, 'trees_mask': trees_mask}


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

        rgb_channels = Image.fromarray(current_npz_frames['A'][:, :, 0:3])
        thermal_channel = Image.fromarray(current_npz_frames['B'][:, :, 0]).convert('RGB')

        # Separate A_seg into one hot matrices for classes of interest.
        # fixme: use correct indices

        # apply the same transform to both rgb, thermal and mask channels
        transform_params = get_params(self.opt, rgb_channels.size)

        input_thermal_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        mask_transform = get_transform(self.opt, transform_params)

        rgb_channels = input_thermal_transform(rgb_channels)
        thermal_channel = input_thermal_transform(thermal_channel)

        return {'rgb_channels': rgb_channels,
                'thermal_channel': thermal_channel,
                'mask_dict': get_mask_dictionary(current_npz_frames['A'][:, :, 3], mask_transform)}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.input_files)
