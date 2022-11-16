set -ex
python train.py --dataroot ../robotrain_pytorch/datasets/FLIR_np --name m_channel --model pix2pix --netG unet_128  --lambda_L1 100 --dataset_mode numpy --norm batch --pool_size 0 --input_nc=5


export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cp $(find ./heappe/direct/set0013/2022-10-20-22553643-People_Anim_1/out/rgb/ -name "*.jpg./heappe/direct/set0013/2022-10-20-22553643-People_Anim_1/out/rgb/" -o -name "*.png") 