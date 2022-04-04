import cv2
import os
import re
import numpy as np

MASK_DIR="F:\\Poles\\Dataset\\Mask\\"
IMAGE_DIR="F:\\Poles\\Dataset\\Image\\"
OUTPUT_DIR="F:\\Poles\\Dataset\\Combine\\"
images = os.listdir(IMAGE_DIR)
masks  = os.listdir(MASK_DIR)
print(images)

def loadImage(path):
    if os.path.exists(path):
        img=cv2.imread(path).astype("float32")
        return img
    else:
       return [0]

def combineMask(paths):
    real_img=cv2.imread(MASK_DIR+paths[0]).astype("float32")
    mask = np.ones(real_img.shape, dtype=np.uint8)
    mask.fill(255)
    for path in paths:
        img = cv2.imread(MASK_DIR+path).astype("float32")
        img[img==255]=0
        mask=mask+img
    mask[mask>255]-=255
    return mask



for image in images:
    id=image.split(".")[0]
    print(id)
    image_color=loadImage(IMAGE_DIR+image)

    mask = np.ones(image_color.shape, dtype=np.uint8)
    mask.fill(255)

    files=[x for x in masks if x.startswith(str(id)+"_1_")]
    if len(files)!=0:
        mask_one=combineMask(files)
        mask_one[mask_one==255]=0

    files=[x for x in masks if x.startswith(str(id)+"_2_")]
    mask_two=[0]
    if len(files)!=0:
        mask_two=combineMask(files)
        mask_two[mask_two == 255] = 0

    files=[x for x in masks if x.startswith(str(id)+"_3_")]
    mask_three=[0]
    if len(files)!=0:
        mask_three=combineMask(files)
        mask_three[mask_three == 255] = 0


    result=mask+mask_one+mask_two+mask_three
    result[result>255]-=255
    cv2.imwrite(OUTPUT_DIR+str(id)+".jpg", result)






