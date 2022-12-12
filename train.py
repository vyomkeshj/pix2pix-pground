import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
# from data.test_loader import TestDataset
from data.validation_loader import ValidationDataset

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    test_dataset = ValidationDataset('./validation_npz')
    # test_dataset = TestDataset('../robotrain_pytorch/datasets/FLIR_np/test')
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    test_size = len(test_dataset)
    dataset_size = len(dataset)    # get the number of images in the dataset.

    train_gen_every = 2 # train generator every even epoch only

    test_every_steps = opt.batch_size*20 # test every epoch and publish validation results
    # viz_sample_every_steps = opt.batch_size * 2 # test every n epochs and publish validation results

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            # model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            model.forward()
            model.optimize_discriminator()

            if i % train_gen_every == 0:
                model.forward()
                model.optimize_generator()

            if total_iters % test_every_steps == 0:
                # upload training gen sample
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch)

                for t_i, test_data in enumerate(test_dataset):
                    if t_i>=test_size:
                        break
                    model.set_input(test_data)  # unpack data from data loader
                    model.test()           # run inference

                    visuals = model.get_current_visuals()  # get image results
                    visualizer.display_current_results(visuals, epoch, is_val=True)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size

                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
