import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from data.validation_loader import ValidationDataset

import wandb

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.

    dataset = ValidationDataset('./validation_npz')
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    for i, data in enumerate(dataset):

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

        visuals = model.get_current_visuals()  # get image results
        visualizer.display_current_results(visuals, i, is_val=True)

