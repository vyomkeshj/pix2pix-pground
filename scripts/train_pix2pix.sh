eval "$(conda shell.bash hook)"
conda activate /scratch/project/open-20-15/envs/pix2pix_env

cd /scratch/project/open-20-15/robotrain
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python train.py --netG=unet_512 --dataroot ../robotrain_data/ --name test_threshold --ndf=100 --ngf=64 --batch_size=8 --n_layers_D=3 --lr=0.00005