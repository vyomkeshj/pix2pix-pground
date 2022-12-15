import numpy as np
import os
import sys
import time
from PIL import Image
from . import util

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. error.')


class Visualizer:
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.name = opt.name
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.save_path = opt.output_dir
        self.use_wandb = opt.use_wandb

        self.image_height = opt.image_height
        self.image_width = opt.image_width

        self.current_epoch = 0
        self.image_index = 0

        if self.use_wandb:
            self.wandb_project_name = opt.wandb_project_name
            self.wandb_run = wandb.init(project=self.wandb_project_name, name=opt.name,
                                    config=opt) if not wandb.run else wandb.run
            self.wandb_run._label(repo='robotrain_inference')

    def reset(self):
        self.current_epoch = 0

    def save_generated_thermal(self, visuals, index):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        generated_thermal_val = util.tensor2im(visuals['generated_thermal'])[0]
        generated_thermal_val[generated_thermal_val >= 250] = np.mean(generated_thermal_val)

        # Thermo highlight using the masks:
        person_mask_val = util.tensor2im(visuals['person_mask'])[:, :, 0]
        np.putmask(generated_thermal_val, (person_mask_val == 255), 245)

        car_mask_val = util.tensor2im(visuals['car_mask'])[:, :, 0]
        np.putmask(generated_thermal_val, (car_mask_val == 255), 245)

        van_mask_val = util.tensor2im(visuals['van_mask'])[:, :, 0]
        np.putmask(generated_thermal_val, (van_mask_val == 255), 245)

        animal_mask_val = util.tensor2im(visuals['animal_mask'])[:, :, 0]
        np.putmask(generated_thermal_val, (animal_mask_val == 255), 245)

        railroad_mask = util.tensor2im(visuals['railroad_mask'])[:, :, 0]
        np.putmask(generated_thermal_val, (railroad_mask == 255), 100)

        image_pil = Image.fromarray(generated_thermal_val)
        # Resize back

        image_pil = image_pil.resize((self.image_width, self.image_height), Image.ANTIALIAS)
        image_pil.save(f"{self.save_path}/thermal_{index}.png")

    def display_current_results(self, visuals, epoch, is_val=False):
        """Display current results on wandb

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            is_val (bool) - - if true, appends _val to image label
        """

        ims_dict = {"epoch": epoch}
        temp_dict = {}
        for label, image in visuals.items():
            label = label + f"_val" if is_val else label
            multiplier = 1
            if "mask" in label:
                multiplier = 100

            if "person_mask_val" in label:
                temp_dict['person_mask_val'] = util.tensor2im(image)

            if "generated_thermal_val" in label:
                temp_dict['generated_thermal_val'] = util.tensor2im(image)

            if "generated" in label:
                image = image[:, 0, :, :]

            image_numpy = util.tensor2im(image * multiplier)

            wandb_image = wandb.Image(image_numpy)
            ims_dict[label] = wandb_image

        if is_val:
            thermal = temp_dict['generated_thermal_val'][0]
            # clean extra bright pixels, replace them with mean (todo: improve)
            thermal[thermal >= 250] = np.mean(thermal)
            person_mask = temp_dict['person_mask_val'][:, :, 0]
            np.putmask(thermal, (person_mask == 255), 245)

            ims_dict['highlighted_thermal_val'] = wandb.Image(thermal[np.newaxis, ...])
        self.wandb_run.log(ims_dict)

    def plot_current_losses(self, losses):
        """display the current losses on wandb display
        Parameters:
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        self.wandb_run.log(losses)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
