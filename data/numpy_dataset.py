import os
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


def create_mask(mask_matrix, id2label_id):
    """ Takes a mask matrix and a label id, returns a boolean mask with just the areas with the label id """
    mask = np.ones_like(mask_matrix) * id2label_id
    mask = 1 * np.equal(mask, mask_matrix)
    return mask[...].astype(np.uint8)


def get_mask_dictionary(segementation_channel, mask_transform):
    trees_mask = create_mask(segementation_channel, 104)
    person_mask = create_mask(segementation_channel, 127)

    trees_mask = Image.fromarray(trees_mask).convert('RGB')
    person_mask = Image.fromarray(person_mask).convert('RGB')

    trees_mask = np.array(mask_transform(trees_mask))[0, :, :][np.newaxis, ...]
    person_mask = np.array(mask_transform(person_mask))[0, :, :][np.newaxis, ...]

    return {'person_mask': person_mask, 'trees_mask': trees_mask}


class NumpyDataset(BaseDataset):
    """A dataset class to load data from a folder with .npz files [RGB+M_seg]

    It assumes that the directory '/path/to/data/train' contains npz files.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB_numpy = np.load(AB_path)

        A_RGB = Image.fromarray(AB_numpy['A'][:, :, 0:3])
        B = Image.fromarray(AB_numpy['B'][:, :, 0]).convert('RGB')

        # Separate A_seg into one hot matrices for classes of interest.
        # fixme: use correct indices

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A_RGB.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        mask_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        # trees = Mask_transform(trees_mask)
        A = A_transform(A_RGB)
        B = B_transform(B)

        return {'A': A,
                'M_dict': get_mask_dictionary(AB_numpy['A'][:, :, 3], mask_transform),
                'B': B,
                'A_paths': AB_path,
                'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
