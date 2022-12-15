eval "$(conda shell.bash hook)"
conda activate /scratch/project/open-20-15/envs/pix2pix_env

# The path on karolina the project is checked out on
cd /scratch/project/open-20-15/robotrain
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# To run Model Variant 1
python3 test.py --name rgb_p2p --model pix2pix --netG unet_256 --dataroot ./validation_dataroot --ngf=64 --ndf=64 --version=rgb --output_dir=./variant_1_result

# To run Model Variant 2
#python3 test.py --name test_threshold --model pix2pix --netG unet_512 --dataroot ./validation_dataroot --version=w_seg --input_nc=7 --output_dir=./variant_2_result
