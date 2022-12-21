eval "$(conda shell.bash hook)"
conda activate /scratch/project/open-20-15/envs/pix2pix_env

cd /scratch/project/open-20-15/robotrain
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# dataroot is the root path with a ./train folder for the .npz files [i.e. ./dataroot/train has .npz files]
python train.py --netG=unet_512 --dataroot ./npz_vesselseg --name renorm --ndf=16 --ngf=16 --batch_size=4 --n_layers_D=3 --lr=0.0002 --n_epochs=10 --n_epochs_decay=90 --train_dis_every=2 --gpu_ids=0