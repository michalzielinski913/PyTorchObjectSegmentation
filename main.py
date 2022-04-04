import torch
import os

from Dataset import SegmentationDataset

print(torch.cuda.get_device_name(0))

IMAGE_PATH="F:\\Poles\\Dataset\\Image\\"
MASK_PATH="F:\\Poles\\Dataset\\Mask\\"
TRAIN_RATIO=0.8

files=os.listdir(IMAGE_PATH)

train_size=int(TRAIN_RATIO*len(files))

train_dataset=SegmentationDataset(IMAGE_PATH, MASK_PATH, files[0:train_size])
test_dataset=SegmentationDataset(IMAGE_PATH, MASK_PATH, files[train_size:])
