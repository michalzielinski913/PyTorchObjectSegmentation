# import the necessary packages
from torch.utils.data import Dataset
import cv2
import os
class SegmentationDataset(Dataset):
	def __init__(self, imageFolder, maskFolder, files):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePath = imageFolder
		self.maskPath = maskFolder
		self.files=files
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.files)
	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.files[idx]
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(self.imagePath+imagePath)
		mask = cv2.imread(self.maskPath+imagePath, 0)
		# check to see if we are applying any transformations
		# return a tuple of the image and its mask
		return (image, mask)