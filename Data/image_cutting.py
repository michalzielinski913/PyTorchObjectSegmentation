import random
import shutil

import numpy as np
from tqdm import tqdm
import os
import cv2
import pickle
from config import weights

class ImageGenerator:
    def get_cut_coordinates(self, img, size=(1024, 1024)):
        """
        Calculate ROI which should be cutted
        :param img: Img from which ROI will be extracted
        :param size: tuple which shows expected ROI size, default (512, 512)
        :return: ROI coordinates in format (start_x, start_y), (end_x, end_y)
        """
        input_size=img.shape
        max_offset=(input_size[0]-size[0], input_size[1]-size[1])
        start_point=(random.randint(0, max_offset[0]), random.randint(0, max_offset[1]))
        end_point=(start_point[0]+size[0], start_point[1]+size[1])
        return start_point, end_point

    def cut_img(self, img, start, stop):
        """
        Extract ROI from given image
        :param img: Image from which ROI will be extracted
        :param start: Start position tuple
        :param stop: End position tuple
        :return: image fragment based on given coordinates
        """
        return img[start[0]:stop[0], start[1]:stop[1],:]

    def evaluate_score(self, mask):
        """
        Calculate image score
        :param mask: Mask which will be used to calculate final score
        :param start: Start position tuple
        :param stop: End position tuple
        :return: Numerical value which represents score of extracted ROI
        """
        classes=mask.shape[2]
        score=0.0
        for img_class in range(classes):
            mask_channel=mask[:,:,img_class]
            weight=1-weights[img_class]
            score+=np.count_nonzero(mask_channel)/mask_channel.size
        return score

    def select_images(self, scores):
        """
        Estimates which indexes should be used
        :param scores: List of scores of each image part
        :return: indexes of choosen images
        """
        indexes=(np.argsort(scores))
        return indexes[0:5]

def split_and_save(source_img, source_mask, destination):
    files = os.listdir(source_img)
    gen=ImageGenerator()
    sizes=[]
    if(os.path.exists(destination)):
        pass
    else:
        os.makedirs(destination)
    for file in tqdm(files):
        img=cv2.imread(source_img+"\\"+file)
        with open(source_mask +"\\"+ file.replace("jpg", "png"), "rb") as f_in:
            mask = pickle.load(f_in)
        mask_list=[]
        img_list=[]
        score_list=[]
        # for i in range(20):
        #     rand=random.uniform(0, 1)
        #     if rand<=0.7:
        #         size=(1024, 1024)
        #     elif rand <0.85:
        #         size=(512, 512)
        #     else:
        #         size=(1536, 1536)
        #     start, stop=gen.get_cut_coordinates(img, size)
        #     new_img=gen.cut_img(img, start, stop)
        #     new_mask=gen.cut_img(mask, start, stop)
        #     mask_list.append(new_mask)
        #     img_list.append(new_img)
        #     score_list.append(gen.evaluate_score(new_mask))
        #for choosen_file_indexes in gen.select_images(score_list):
        for idx in range(4):

            image = np.rot90(img, idx).copy()
            masks = np.rot90(mask, idx).copy()
            cv2.imwrite(destination+"\\"+str(idx)+"_"+file, image)
            with open(destination+"\\m_"+str(idx)+"_"+file, "wb") as f_out:
                pickle.dump(masks, f_out)

def prepare_test_set(mask_path, output):
    files = os.listdir(mask_path)
    for file in files:
        new_file="m_"+file.replace(".png", ".jpg")
        os.rename(mask_path+file, mask_path+new_file)
        shutil.copy(mask_path+new_file, output+new_file)



if __name__=="__main__":
    split_and_save("C:\Dataset\Split\Train\IMG", "C:\Dataset\Split\Train\MASK", "C:\Dataset\SplitWeight\Train")
    split_and_save("C:\Dataset\Split\Validation\IMG", "C:\Dataset\Split\Validation\MASK", "C:\Dataset\SplitWeight\Validation")
    prepare_test_set("C:\Dataset\Split\Test\MASK\\", "C:\Dataset\Split\Test\\")