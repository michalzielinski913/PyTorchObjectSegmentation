import os
import shutil
from tqdm import tqdm
from config import TEST_IMAGES_FILENAMES, TRAIN_RATIO

files = os.listdir("G:\Dataset\Image2\\")
files=[item for item in files if item not in TEST_IMAGES_FILENAMES]

##Copying test images

for test_image in TEST_IMAGES_FILENAMES:
    path="G:\Dataset\Image2\\"+test_image
    shutil.copy(path, "G:/Dataset/Split/Test")
    path=path.replace("jpg", "png").replace("Image2", "Combine2")
    shutil.copy(path, "G:/Dataset/Split/Test/MASK")
train_size = int(TRAIN_RATIO * len(files))

for train_file in tqdm(files[0:train_size]):
    path="G:\Dataset\Image2\\"+train_file
    shutil.copy(path, "G:/Dataset/Split/Train/IMG")
    path=path.replace("jpg", "png").replace("Image2", "Combine2")
    shutil.copy(path, "G:/Dataset/Split/Train/MASK")

for val_file in tqdm(files[train_size:]):
    path="G:\Dataset\Image2\\"+val_file
    shutil.copy(path, "G:/Dataset/Split/Validation/IMG")
    path=path.replace("jpg", "png").replace("Image2", "Combine2")
    shutil.copy(path, "G:/Dataset/Split/Validation/MASK")