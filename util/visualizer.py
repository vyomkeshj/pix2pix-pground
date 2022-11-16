import numpy as np
import os
import sys
import ntpath
import time
from . import util
from subprocess import Popen, PIPE


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. error.')


def save_images(visuals):
    """Save images to the wandb.

    Parameters:
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
    """
    ims_dict = {}
    for label, im_data in visuals.items():
        print("input image = "+f"label {label}"+str(im_data.shape))
        imgs = util.tensor2im(im_data)
        gen_thermal = imgs[0].shape
        print("converted image:: "+str(gen_thermal))

        for i, im in enumerate(imgs):

            ims_dict[label] = wandb.Image(im)
        wandb.log(ims_dict)


class Visualizer():
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


        self.use_wandb = True
        self.wandb_project_name = opt.wandb_project_name

        self.wandb_run = wandb.init(project=self.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        self.wandb_run._label(repo='CycleGAN-and-pix2pix')
        self.current_epoch = 0

        #
        # # create a logging file to store training losses
        # self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        # with open(self.log_name, "a") as log_file:
        #     now = time.strftime("%c")
        #     log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, epoch):
        """Display current results on wandb

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """

        # if self.use_wandb:
        columns = [key for key, _ in visuals.items()]
        columns.insert(0, 'epoch')
        result_table = wandb.Table(columns=columns)
        table_row = [epoch]
        ims_dict = {}
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            wandb_image = wandb.Image(image_numpy)
            table_row.append(wandb_image)
            ims_dict[label] = wandb_image
        self.wandb_run.log(ims_dict)
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            result_table.add_data(*table_row)
            self.wandb_run.log({"Result": result_table})

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
