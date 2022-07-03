import os

import segmentation_models_pytorch as smp
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from config import DETECTION_THRESHOLD, ID_TO_NAME, NUM_CLASSES
import pandas as pd
import seaborn as sn
import pandas as pd

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
    x=max(int(len(train_loss)/2), 5)
    plt.figure(figsize=(x, 5))
    plt.plot(epochs, train_loss, 'b', label='Training loss')


    plt.plot(epochs, val_loss, '#FFA500', label='Validation loss')
    plt.xticks(epochs)
    plt.legend(loc="upper right")

    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(path)
    plt.close()

def generate_class_loss_plot(path, losses):
    """
    Generate plot of train and validation loss and store it in a given location
    :param path: where data will be stored
    :param losses: List of loss values
    """

    epochs = [*range(0,len(losses))]
    losses=list(map(list, zip(*losses))) #Transpose
    x=max(int(len(epochs)/2),10)
    plt.figure(figsize=(x, 12))
    for class_id, class_losses in enumerate(losses):
        plt.plot(epochs, class_losses, label='Class: {}, {}'.format(class_id, ID_TO_NAME[class_id]))

    plt.xticks(epochs)
    plt.legend(loc="upper right")

    plt.title('Class loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(path)
    plt.close()


def visualize(filename, label, **images):
    """
    Store predicted mask next to real one
    :param filename:
    :param label
    :param images:
    :return:
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title()+" "+ID_TO_NAME[label])
        if image.shape[0]==3:
            image=np.rollaxis(image, 0, 3)
        # image[image >= DETECTION_THRESHOLD] = 255
        # image[image < DETECTION_THRESHOLD] = 0
        plt.imshow(image)
    plt.savefig(filename)
    plt.close()

def visualize(filename, Image, Prediction, RealMask):
    """
    Store predicted mask next to real one in single image for all classes
    :param filename:
    :param label
    :param images:
    :return:
    """

    plt.figure(figsize=(16, 55))
    columns = 3
    rows = NUM_CLASSES
    x=0
    for i in range(1, columns * rows + 1,3):
        plt.add_subplot(rows, columns, i)
        plt.imshow(Image[x])
        plt.add_subplot(rows, columns, i+1)
        plt.imshow(Prediction[x])
        plt.add_subplot(rows, columns, i+2)
        plt.imshow(RealMask[x])
        x+=1
    plt.show()
    plt.savefig(filename)
    plt.close()

def metrics_calculation(y_pred, y_true):
    """
    Calculate metrics for given predictions
    :param y_pred: True masks
    :param y_true: Predicted masks
    :return: iou score, f1 score, f2 score, accuracy, recall
    """
    out = (y_pred > DETECTION_THRESHOLD).float()
    input=y_true.to(torch.int32)
    out=out.to(torch.int32)
    tp, fp, fn, tn =smp.metrics.functional.get_stats(out, input, mode='multilabel')
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
    return iou_score.cpu().detach().item(), f1_score.cpu().detach().item(), f2_score.cpu().detach().item(), accuracy.cpu().detach().item(), recall.cpu().detach().item()

def _res_eval(x, y):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(11):
        for j in range(512):
            for k in range(512):
                if round(x[i][j][k][0]) == True and y[i][j][k][0] == True:
                    TP = TP + 1
                elif round(x[i][j][k][0]) == False and y[i][j][k][0] == False:
                    TN = TN + 1
                elif round(x[i][j][k][0]) == True and y[i][j][k][0] == False:
                    FP = FP + 1
                elif round(x[i][j][k][0]) == False and y[i][j][k][0] == True:
                    FN = FN + 1
    return TP, TN, FP, FN

def test_matrix(y_pred, y_true):
    y_pred=y_pred[0].cpu().data.numpy()
    y_pred[y_pred>=DETECTION_THRESHOLD]=1.
    y_pred[y_pred<DETECTION_THRESHOLD]=0.
    y_pred=y_pred.astype(int)
    y_true=y_true[0].cpu().data.numpy().astype(int)
    #TP, FN, FP, TN=


def confusion_matrix(path, y_pred, y_true):
    y_pred = y_pred[0].cpu().data.numpy()
    y_true = y_true[0].cpu().data.numpy()
    y_p=DETECTION_THRESHOLD<=y_pred
    y_t=DETECTION_THRESHOLD<=y_true
    Matrix = [[0 for x in range(NUM_CLASSES)] for y in range(NUM_CLASSES)]

    for i in range(NUM_CLASSES):
        sum=1.0
        for x in range(NUM_CLASSES):
            res=np.logical_and(y_p[i], y_t[x])
            count = np.count_nonzero(res)/max(np.count_nonzero(y_p[i]),1)
            Matrix[i][x]=count
            sum=sum-count
            if count>=1:
                Matrix[i][x]=0
        if np.count_nonzero(y_t[i]) == 0:
            Matrix[i][i] = np.count_nonzero(y_p[i]==0)/max(np.count_nonzero(y_t[i]==0), 1)
    df_cm = pd.DataFrame(Matrix, index = ID_TO_NAME.values(),
                      columns =ID_TO_NAME.values())
    plt.figure(figsize = (15,10))
    plot=sn.heatmap(df_cm, annot=True,vmin=0, vmax=1)
    fig = plot.get_figure()

    fig.savefig(path)
    plt.close()

def generate_plot(train_csv_path, val_csv_path, metric, save_location):
    """
    Generate train validation plot for single metric
    :param train_csv_path: Path to csv file with train data
    :param val_csv_path: Path to csv file with validation data
    :param metric: Which metric should be checked
    :param save_location: Where to store generated plot
    """
    train_df=pd.read_csv(train_csv_path, sep=";")
    val_df=pd.read_csv(val_csv_path, sep=";")
    train_data_df=train_df[['epoch', metric]]
    val_data_df=val_df[['epoch', metric]]
    epochs=int(train_df[['epoch']].iloc[-1])
    train_list=[]
    val_list=[]
    for epoch in range(epochs+1):
        train_epoch=train_data_df.loc[train_data_df['epoch']==epoch]
        val_epoch=val_data_df.loc[val_data_df['epoch']==epoch]
        train_list.append(train_epoch[metric].mean())
        val_list.append(val_epoch[metric].mean())

    epochs = [*range(0,len(val_list))]
    x=max(int(len(epochs)/2),10)
    plt.figure(figsize=(x, 12))

    plt.plot(epochs, train_list, label='Train {}'.format(metric))
    plt.plot(epochs, val_list, label='Validation {}'.format(metric))

    plt.xticks(epochs)
    plt.legend(loc="upper right")

    plt.title('{} loss'.format(metric))
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.savefig(save_location)
    plt.close()
