# Approximate Thermal image from RGB+Mask

### To run training and to create the model

1. Clone the repository on Karolina
2. Run `./scripts/train.sh` 

This will load the data from `/scratch/project/open-20-15/robotrain_data` and train the model.
The model will be saved in `./checkpoints`

### To run inference using this model

1. Run `create_validation_dataset.py` after editing it with the correct paths for the data. It will convert the image+mask pair into .npz format that the test script will be able to load and perform inference on.

2. Run `./scripts/test.sh` with the correct dataroot, you specify dataroot in `create_validation_dataset.py`

