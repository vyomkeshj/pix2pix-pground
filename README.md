# Approximate Thermal image from RGB+Mask

### To run training and to create the model

1. Clone the repository on Karolina
2. Run `./scripts/train.sh` 

This will load the data from `/scratch/project/open-20-15/robotrain_data` and train the model.
The model will be saved in `./checkpoints`

### To run inference using this model

1. Run `create_validation_dataset.py --<input_directory> --<validation_dataroot>`
It will convert the image+mask pair into .npz format that the test script will be able to load and perform inference on.
    <b>Example for input_directory is the `select_validation` folder, it has both image and corresponding mask</b>

2. Run `./scripts/test.sh` with the correct `--dataroot=<validation_dataroot>` and `--output_dir=<path_for_generated_thermal>`

Note: for test.sh to work, you must ensure that the correct model is in the `./checkpoints/` directory of the repo clone

