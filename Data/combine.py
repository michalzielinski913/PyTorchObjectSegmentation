import cv2
import os
import re
import pickle

import numpy as np
from tqdm import tqdm
MASK_DIR="F:\\Poles\\Dataset\\Mask2\\"
IMAGE_DIR="F:\\Poles\\Dataset\\Image2\\"
OUTPUT_DIR="F:\\Poles\\Dataset\\Combine2\\"
images = os.listdir(IMAGE_DIR)
masks  = os.listdir(MASK_DIR)

def loadImage(path):
    if os.path.exists(path):
        img=cv2.imread(path).astype("uint8")
        return img
    else:
       return [0]

def combineMask(paths):
    real_img=cv2.imread(MASK_DIR+paths[0]).astype("uint8")
    mask = np.ones(real_img.shape, dtype=np.uint8)
    mask.fill(255)
    #Zbadać typ danych w zdjęciu (float?)
    for file in paths:
        img=cv2.imread(MASK_DIR+file)
        mask+=img

    mask[mask==np.amin(mask)]=0

    return mask

image_labels=["_1_", "_2", "_3_", "_4_", "_5_", "_7_", "_8_", "_9_", "_10_", "_11_", "_12_"]
#images=["101.jpg"]
for image in tqdm(images):
    id=image.split(".")[0]

    image_color=loadImage(IMAGE_DIR+image)

    mask = np.ones((image_color.shape[0], image_color.shape[1], 11), dtype=np.uint8)
    mask.fill(0)
    result=[]
    for index, label in enumerate(image_labels):
        files=[x for x in masks if x.startswith(str(id)+label)]
        if len(files)!=0:
            mask_tmp=combineMask(files)
            mask[:,:,index]=(mask_tmp[:,:,0])

    result=np.array(result)

    mask[mask>np.amin(mask)]=255
    unique, counts = np.unique(mask, return_counts=True)
    print(list(zip(unique, counts)))
    #result=cv2.bitwise_not(result)
    with open(OUTPUT_DIR+str(id)+".png", "wb") as f_out:
        pickle.dump(mask, f_out)
    #cv2.imwrite(OUTPUT_DIR+str(id)+".png", mask)






