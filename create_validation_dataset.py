import numpy as np
import os
import re
from PIL import Image
import argparse

if __name__ == '__main__':
    """Reads a set of images and their corresponding mask from the input_directory,
     converts it into .npz format supported by the model data loader
     
     Example: input_directory uses the naming convention: [gt4400.png, rgb4400.png, gtxxx.png, rgbyyy.png, ..] for files
     """

    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", type=str, required=True,  help="The path to the input directory")
    parser.add_argument("validation_dataroot", required=True, type=str, help="output for .npz")

    args = parser.parse_args()

    input_directory = args.input_directory
    validation_dataroot = args.validation_dataroot

    # # Example:
    # input_directory = './select_validation'
    # # directory where the output .npz files will be saved, you will pass this as argument in `scripts/test_pix2pix.sh`
    # validation_dataroot = './validation_dataroot'

    # create a list to store the image pairs
    image_pairs = []

    # get a list of all the files in the directory
    files = os.listdir(input_directory)

    # loop through the files, load each image pair, save as npz
    # save_index = 0
    for file in files:
        if file.endswith('.png'):
            if 'rgb' in file:
                index = re.search('\d+', file).group(0)
                rgb_image = np.array(Image.open(os.path.join(input_directory, file)))[:, :, 0:3]
                mask = np.array(Image.open(os.path.join(input_directory, f"gt{index}.png")))[:, :, 0]
                stacked_array = np.stack([mask, mask, mask], axis=2)

                stacked = np.dstack((rgb_image, mask))

                empty_thermal = (rgb_image[:, :, 0])[..., np.newaxis]
                empty_thermal_sh = np.zeros_like(stacked_array)[:, :, 0][..., np.newaxis]

                np.savez_compressed(f'{validation_dataroot}/{index}', A=stacked, B=empty_thermal)
                # save_index += 1
