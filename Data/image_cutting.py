import random

def get_cut_coordinates(img, size=(1024, 1024)):
    """
    Calculate ROI which should be cutted
    :param img: Img from which ROI will be extracted
    :param size: tuple which shows expected ROI size, default (1024, 1024)
    :return: ROI coordinates in format (start_x, start_y), (end_x, end_y)
    """
    input_size=img.shape
    max_offset=(input_size[0]-size[0], input_size[1]-size[1])
    start_point=(random.randint(0, max_offset[0]), random.randint(0, max_offset[1]))
    end_point=(start_point[0]+size[0], start_point[1]+size[1])
    return start_point, end_point

def cut_img(img, start, stop):
    """
    Extract ROI from given image
    :param img: Image from which ROI will be extracted
    :param start: Start position tuple
    :param stop: End position tuple
    :return:
    """
    return img[start[0]:stop[0], start[1]:stop[1],:]