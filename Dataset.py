# import the necessary packages
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import pickle
from skimage.transform import resize


import cv2
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
class SegmentationDataset(Dataset):
	def __init__(self, imageFolder, maskFolder, files, transform=None):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePath = imageFolder
		self.maskPath = maskFolder
		self.files=files
		self.transforms=transform
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.files)
	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.files[idx]

		image = cv2.imread(self.imagePath+imagePath)
		imagePath = imagePath.replace("jpg", "png")
		with open(self.maskPath+imagePath, "rb") as f_in:
			mask = pickle.load(f_in)

		if self.transforms is not None:
			image = self.transforms(image)
			mask = self.transforms(mask)
		return (image, mask)