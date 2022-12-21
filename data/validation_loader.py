import albumentations as alb
import numpy as np
import os
import torch
from PIL import Image

from data.utils import load_frames, get_transform


def get_transformed_images_masks(input_image, seg_channel, thermal_image, transform):
    rgb_channels = input_image[:, :, 0:3]

    # get one hot mask by R channel value of the RGB seg mask
    trees_mask = create_mask(seg_channel, [97])
    person_mask = create_mask(seg_channel, [102])
    railroad_mask = create_mask(seg_channel, [104, 106])
    sky_mask = create_mask(seg_channel, [0])
    van_mask = create_mask(seg_channel, [203])
    car_mask = create_mask(seg_channel, [88])
    animal_mask = create_mask(seg_channel, [120])

    thermal_stack = np.dstack([thermal_image, thermal_image, thermal_image])

    transformed = transform(image=rgb_channels,
                            thermal_image=thermal_stack,
                            trees_mask=trees_mask,
                            person_mask=person_mask,
                            railroad_mask=railroad_mask,
                            sky_mask=sky_mask,

                            van_mask=van_mask,
                            car_mask=car_mask,
                            animal_mask=animal_mask)

    return transformed['image'], (transformed['thermal_image'][:, :, 0]), \
        {
            'person_mask': transformed['person_mask'][..., np.newaxis],
            'trees_mask': transformed['trees_mask'][..., np.newaxis],
            'railroad_mask': transformed['railroad_mask'][..., np.newaxis],
            'sky_mask': transformed['sky_mask'][..., np.newaxis],
            # masks the model does not take as input
            'van_mask': transformed['van_mask'][..., np.newaxis],
            'car_mask': transformed['car_mask'][..., np.newaxis],
            'animal_mask': transformed['animal_mask'][..., np.newaxis]
        },


class ValidationDataset:

    def __init__(self, path):
        self.data_path = path  # os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.input_files = sorted(load_frames(self.data_path, float("inf")))  # get image paths

    def __getitem__(self, index):
        filename = self.input_files[index]
        current_npz_frames = np.load(filename)

        rgb_channels = current_npz_frames['A'][:, :, 0:3]
        mask_channel = current_npz_frames['A'][:, :, 3]
        thermal_channel = current_npz_frames['B'][:, :, 0]
        transform = alb.Compose([
            alb.Resize(width=512, height=512, interpolation=1, always_apply=True),
            # alb.RandomCrop(width=512, height=512),
            alb.CLAHE(clip_limit=2.0, tile_grid_size=(3, 3), always_apply=True),
        ], additional_targets={
            'image': 'image',
            'thermal_image': 'image',
            'person_mask': 'mask',
            'trees_mask': 'mask',
            'railroad_mask': 'mask',
            'sky_mask': 'mask',

            'van_mask': 'mask',
            'car_mask': 'mask',
            'animal_mask': 'mask'})

        transformed_image, transformed_thermal, mask_dict = get_transformed_images_masks(
            rgb_channels,
            mask_channel,
            thermal_channel,
            transform)

        normalizer = get_transform()
        transformed_image = normalizer(transformed_image)
        transformed_image = torch.permute(transformed_image, (1, 2, 0))

        mask_dict = dict(map(lambda item: (item[0], torch.tensor(item[1][np.newaxis, ...])), mask_dict.items()))
        return {'rgb_channels': torch.tensor(transformed_image[np.newaxis, ...]),
                'thermal_channel': torch.tensor(transformed_thermal[np.newaxis, ...]),
                'mask_dict': mask_dict}, filename

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.input_files)
