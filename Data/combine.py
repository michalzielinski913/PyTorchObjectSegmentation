import cv2
import os
import re
import numpy as np

MASK_DIR="F:\\Poles\\Dataset\\Mask\\"
IMAGE_DIR="F:\\Poles\\Dataset\\Image\\"
OUTPUT_DIR="F:\\Poles\\Dataset\\Combine\\"
images = os.listdir(IMAGE_DIR)
masks  = os.listdir(MASK_DIR)

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

image_labels=["_1_", "_2_", "_5"]

for image in images:
    id=image.split(".")[0]
    print(id)
    image_color=loadImage(IMAGE_DIR+image)

    mask = np.ones(image_color.shape, dtype=np.uint8)
    mask.fill(255)
    result=mask
    for label in image_labels:
        files=[x for x in masks if x.startswith(str(id)+label)]
        if len(files)!=0:
            mask_tmp=combineMask(files)
            mask_tmp[mask_tmp==255]=0
            result+=mask_tmp

    result[result>255]-=255
    cv2.imwrite(OUTPUT_DIR+str(id)+".jpg", result)






