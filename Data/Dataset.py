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
    def __init__(self, imageFolder, transform=None):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePath = imageFolder
        self.transforms = transform
        self.files=[]
        for file in os.listdir(self.imagePath):
            if not (file.startswith("m_")):
                self.files.append(file)


    def __len__(self):
        # return the number of total samples contained in the dataset
        return 4 * len(self.files)

    def __getitem__(self, idx):
        # grab the image path from the current index
        index = idx // 4
        imagePath = self.files[index]

        image = cv2.imread(self.imagePath + imagePath)

        with open(self.imagePath + "m_" + imagePath, "rb") as f_in:
            mask = pickle.load(f_in)

        rotation = idx % 4
        image = np.rot90(image, rotation).copy()
        mask = np.rot90(mask, rotation).copy()
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return (image, mask)


class Detection(Dataset):
    def __init__(self, files, transform=None):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.files = files
        self.transforms = transform

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.files)

    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.files[idx]
        image = cv2.imread(imagePath)
        if self.transforms is not None:
            image = self.transforms(image)

        return (image, image)

