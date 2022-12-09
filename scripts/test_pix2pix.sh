set -ex
python test.py --dataroot ../robotrain_pytorch/datasets/FLIR_np --name test_model --model pix2pix --netG unet_128  --dataset_mode numpy --input_nc=7 --output_nc=1