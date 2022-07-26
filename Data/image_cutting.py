import random
import numpy as np
from tqdm import tqdm
class ImageGenerator:
    def get_cut_coordinates(self, img, size=(512, 512)):
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
        weight=0.0
        for img_class in range(classes):
            mask_channel=mask[:,:,img_class]
            score+=np.count_nonzero(mask_channel)/mask_channel.size
            if np.count_nonzero(mask_channel)>0:
                weight+=1/classes
        return weight*score

    def select_images(self, scores):
        """
        Estimates which indexes should be used
        :param scores: List of scores of each image part
        :return: indexes of choosen images
        """
        indexes=(np.argsort(scores))
        return [indexes[-1], indexes[-2], indexes[-3], indexes[-9], indexes[-10]]

if __name__=="__main__":
    import os
    from config import IMAGE_PATH, MASK_PATH
    import cv2
    import pickle
    files = os.listdir(IMAGE_PATH)
    SAVE_PATH="G:\\Dataset\\Cut\\"
    gen=ImageGenerator()
    for file in tqdm(files):
        img=cv2.imread(IMAGE_PATH+file)
        with open(MASK_PATH + file.replace("jpg", "png"), "rb") as f_in:
            mask = pickle.load(f_in)
        mask_list=[]
        img_list=[]
        score_list=[]
        for i in range(15):
            start, stop=gen.get_cut_coordinates(img)
            new_img=gen.cut_img(img, start, stop)
            new_mask=gen.cut_img(mask, start, stop)
            mask_list.append(new_mask)
            img_list.append(new_img)
            score_list.append(gen.evaluate_score(new_mask))
        for choosen_file_indexes in gen.select_images(score_list):
            cv2.imwrite(SAVE_PATH+"IMG\\"+str(choosen_file_indexes)+"_"+file, img_list[choosen_file_indexes])
            with open(SAVE_PATH+"MASK\\"+str(choosen_file_indexes)+"_"+file, "wb") as f_out:
                pickle.dump(mask_list[choosen_file_indexes], f_out)
            #print("Saving {} with score {:5f}".format(file, score_list[choosen_file_indexes]))

