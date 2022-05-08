import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import DETECTION_THRESHOLD


def getJSON(dir):
    """
    Get all json files in given directory (including sub directories)
    :param dir: Directory where are json files
    :return: List of all json files
    """
    filelist = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if (file.endswith(".json")):
                filelist.append(os.path.join(root, file))
    return filelist

def get_device():
    """
    Checks if CUDA device is available and returns it, CPU otherwise
    :return: device in string form
    """
    if torch.cuda.is_available():
        DEVICE = 'cuda:0'
        print('Running on the GPU')
    else:
        DEVICE = "cpu"
        print('Running on the CPU')
    return DEVICE

def save_masks(path, filename, img, num_classes):
    """
    Save masks for each class in a image
    :param path: output path
    :param filename: name of the file
    :param img: Image
    :param num_classes: how many classes given image has
    """
    for i in range(num_classes):
        mask=mask[:,:,img]
        mask_path="{}{}_{}.jpg".format(path, filename, i)
        cv2.imwrite(mask_path, mask)

def generate_train_val_plot(path, train_loss, val_loss):
    """
    Generate plot of train and validation loss and store it in a given location
    :param path: where data will be stored
    :param train_loss: avg train loss on all epochs
    :param val_loss: avg validation loss on all epochs
    """
    epochs = [*range(0,len(train_loss))]
    plt.figure(figsize=(int(len(train_loss)/2), 5))
    plt.plot(epochs, train_loss, 'b', label='Training loss')


    plt.plot(epochs, val_loss, '#FFA500', label='Validation loss')
    plt.xticks(epochs)
    plt.legend(loc="upper right")

    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(path)
    plt.close()

def visualize(filename, **images):
    """
    Store predicted mask next to real one
    :param filename:
    :param images:
    :return:
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if image.shape[0]==3:
            image=np.rollaxis(image, 0, 3)
        # image[image >= DETECTION_THRESHOLD] = 255
        # image[image < DETECTION_THRESHOLD] = 0
        plt.imshow(image)
    plt.savefig(filename)
    plt.close()