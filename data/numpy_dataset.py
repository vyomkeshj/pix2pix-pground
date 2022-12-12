import os
import numpy as np
from data.base_dataset import BaseDataset
from data.numpy_loader import load_frames
from PIL import Image
import albumentations as alb

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
    
class NumpyDataset(BaseDataset):

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

        mask_channel = current_npz_frames['A'][:, :, 0:3]
        thermal_channel = current_npz_frames['B'][:, :, 0]

        transform = alb.Compose([
            # alb.Rotate (limit=20, interpolation=1, border_mode=4, value=None, mask_value=None, rotate_method='largest_box', crop_border=False, always_apply=True, p=0.3),
            alb.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.4, hue=0.4, always_apply=False, p=0.2),
            alb.RandomCrop(width=512, height=512),
            alb.RGBShift(r_shift_limit=90, g_shift_limit=90, b_shift_limit=90, always_apply=False, p=0.8),
            # alb.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True),
        ], additional_targets={
            'image': 'image',
            'thermal_image': 'mask',
        })

        transformed = transform(image=mask_channel,
                                thermal_image=thermal_channel)

        # print(f"transformed_image shape: {transformed_image.shape}")

        # return {'seg_channel': rgb2gray(transformed['image'])/255.,
        #         'thermal_channel': transformed['thermal_image']/255.}
        
        return {'seg_channel': transformed['image']/255.,
                'thermal_channel': transformed['thermal_image']/255.}
        

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.input_files)

