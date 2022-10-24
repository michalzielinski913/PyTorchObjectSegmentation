import random
import numpy as np
from tqdm import tqdm
import os
import cv2
import pickle
from config import weights



def cut_img(img, mask, size=(512, 512)):
    """
    Extract ROI from given image
    :param img: Image from which ROI will be extracted
    :param start: Start position tuple
    :param stop: End position tuple
    :return: image fragment based on given coordinates
    """
    input_size=img.shape
    X= input_size[0]//size[0]
    Y= input_size[1]//size[1]

    for row in range(X):
        for column in range(Y):
            img_fragment=img[row*size[0]:row*size[0]+size[0],column*size[1]:column*size[1]+size[1],:]
            mask_fragment=mask[row*size[0]:row*size[0]+size[0],column*size[1]:column*size[1]+size[1],:]
            yield img_fragment, mask_fragment

def split_and_save(source_img, source_mask, destination):
    files = os.listdir(source_img)
    if(os.path.exists(destination)):
        pass
    else:
        os.makedirs(destination)
    for file in tqdm(files):
        img=cv2.imread(source_img+"\\"+file)
        with open(source_mask +"\\"+ file.replace("jpg", "png"), "rb") as f_in:
            mask = pickle.load(f_in)
        X=0
        for img_fragment, mask_fragment in cut_img(img, mask):
            cv2.imwrite(destination+"\\"+str(X)+"_"+file, img_fragment)
            with open(destination+"\\m_"+str(X)+"_"+file, "wb") as f_out:
                pickle.dump(mask_fragment, f_out)
            X+=1
if __name__=="__main__":
    split_and_save("C:\Dataset\Split\Train\IMG", "C:\Dataset\Split\Train\MASK", "C:\Dataset\SplitWeight\Train")
    split_and_save("C:\Dataset\Split\Validation\IMG", "C:\Dataset\Split\Validation\MASK", "C:\Dataset\SplitWeight\Validation")
