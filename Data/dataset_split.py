import os
import shutil
from tqdm import tqdm
from config import TEST_IMAGES_FILENAMES, TRAIN_RATIO

files = os.listdir("C:\Dataset\Image2\\")
print(len(files))
files=[item for item in files if item not in TEST_IMAGES_FILENAMES]
print(len(files))
##Copying test images

for test_image in tqdm(TEST_IMAGES_FILENAMES):
    path="C:\Dataset\Image2\\"+test_image
    shutil.copy(path, "C:/Dataset/Split/Test")
    path=path.replace("jpg", "png").replace("Image2", "Combine2")
    shutil.copy(path, "C:/Dataset/Split/Test")
    os.rename("C:/Dataset/Split/Test/"+test_image.replace("jpg", "png"), "C:/Dataset/Split/Test/m_"+test_image)

train_size = int(TRAIN_RATIO * len(files))
#
# for train_file in tqdm(files[0:train_size]):
#     path="C:\Dataset\Image2\\"+train_file
#     shutil.copy(path, "C:/Dataset/Split/Train/IMG")
#     path=path.replace("jpg", "png").replace("Image2", "Combine2")
#     shutil.copy(path, "C:/Dataset/Split/Test/")
# for val_file in tqdm(files[train_size:]):
#     path="C:\Dataset\Image2\\"+val_file
#     shutil.copy(path, "C:/Dataset/Split/Validation/IMG")
#     path=path.replace("jpg", "png").replace("Image2", "Combine2")
#     shutil.copy(path, "C:/Dataset/Split/Validation/MASK")