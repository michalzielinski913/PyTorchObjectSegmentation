import cv2
import os
import re
import numpy as np
from tqdm import tqdm
MASK_DIR="F:\\Poles\\Dataset\\Mask\\"
IMAGE_DIR="F:\\Poles\\Dataset\\Image\\"
OUTPUT_DIR="F:\\Poles\\Dataset\\Combine\\"
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

    for file in paths:
        img=cv2.imread(MASK_DIR+file)
        mask+=img

    mask[mask==np.amin(mask)]=0

    return mask

image_labels=["_1_", "_3_", "_5"]
#images=["101.jpg"]
for image in tqdm(images):
    id=image.split(".")[0]

    image_color=loadImage(IMAGE_DIR+image)

    mask = np.ones(image_color.shape, dtype=np.uint8)
    mask.fill(0)
    result=[]
    for index, label in enumerate(image_labels):
        files=[x for x in masks if x.startswith(str(id)+label)]
        if len(files)!=0:
            mask_tmp=combineMask(files)
            mask[:,:,index]=(mask_tmp[:,:,0])

    result=np.array(result)

    mask[mask>np.amin(mask)]=255
    # unique, counts = np.unique(mask, return_counts=True)
    # print(list(zip(unique, counts)))
    #result=cv2.bitwise_not(result)
    cv2.imwrite(OUTPUT_DIR+str(id)+".png", mask)






