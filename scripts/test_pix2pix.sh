eval "$(conda shell.bash hook)"
conda activate /scratch/project/open-20-15/envs/pix2pix_env

cd /scratch/project/open-20-15/robotrain
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python test.py --name test_threshold --model pix2pix --netG unet_512 --dataroot ./validation_npz_sample