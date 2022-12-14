import os
import wandb

from data import create_dataset
# from data.validation_loader import ValidationDataset
from data.validation_loader_images import ValidationImageDataset
from models import create_model
from options.test_options import TestOptions
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling;
    opt.use_wandb = False

    # dataset = ValidationDataset(opt.dataroot)
    dataset = ValidationImageDataset(path_rgb=opt.rgb_dir, path_mask=opt.seg_dir)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)
    visualizer = Visualizer(opt)
    for i, (data, original_index) in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference

        visuals = model.get_current_visuals()  # get image results
        visualizer.save_generated_thermal(visuals, index = original_index)
        # visualizer.display_current_results(visuals, i, is_val=True) # upload to wandb
