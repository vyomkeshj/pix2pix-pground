set -ex
python test.py --dataroot ../robotrain_pytorch/datasets/FLIR_np --name m_channel --model pix2pix --netG unet_128  --dataset_mode numpy

python test.py --dataroot ../robotrain_pytorch/datasets/FLIR_np --name m_channel --model pix2pix --netG unet_128  --dataset_mode numpy --input_nc 5
