import os
import random
import numpy as np
from tqdm import tqdm
import pickle

from config import NUM_CLASSES

files = os.listdir("G:\Dataset\FullDataset\IMG")
counter=0
weights=[0]*NUM_CLASSES

for file in tqdm(files):
    counter+=1
    filename=file.split("\\")[-1]
    with open("G:\Dataset\FullDataset\MASK\\" + filename.replace("jpg", "png"), "rb") as f_in:
        mask = pickle.load(f_in)
        classes=mask.shape[2]
        for img_class in range(classes):
            mask_channel=mask[:,:,img_class]
            score=np.count_nonzero(mask_channel)/mask_channel.size
            weights[img_class]=(weights[img_class]*(counter-1)+score)/counter
print(weights)
