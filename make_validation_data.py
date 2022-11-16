import os
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import re

from PIL import Image
import segmentation_models as sm
from tqdm import tqdm
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt

device = torch.device("cpu")

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

model.to(device)

masked_images = []
index = 0
# thermal_pairs = []
# for filename in tqdm(os.listdir('./align/AnnotatedImages/')):
for filename in tqdm(os.listdir('../robotrain_pytorch/datasets/FLIR_np/train/')):
    # if np.random.random_sample() > 0.8:
        try:
          input_frame = np.load(filename)
          rgb_channels = Image.fromarray(current_npz_frames['A'][:, :, 0:3])

# vegetation_mask = create_mask(seg_mask, 8)
          # terrain_mask = create_mask(seg_mask, 9)
          # car_mask = create_mask(seg_mask, 13)
          # person_mask = create_mask(seg_mask, 11)
          #
          # # im = Image.fromarray(seg_mask)
          # # im.save(f"./segmasks/{filename}")
          # stacked = np.dstack((rgb_img, vegetation_mask))
          # stacked = np.dstack((stacked, terrain_mask))
          # stacked = np.dstack((stacked, car_mask))
          # stacked = np.dstack((stacked, person_mask))
          #
          # # thermal = Image.open(f"/content/drive/MyDrive/thermal/{filename}")
          # # thermal = np.array(thermal.resize((256, 256))) / 255.
          #
          # # print(stacked.shape)
          # masked_images.append(stacked)
        except:
          print("c")

          # thermal_pairs.append(thermal)


with open(f"./val_data.npz", 'wb') as f:
    np.save(f, masked_images)
# with open(f"/content/drive/MyDrive/merged4/thermal_data_min.npz", 'wb') as f:
#     np.save(f, thermal_pairs)