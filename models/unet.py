import torch
import os
import csv

import torchvision
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from Data.Dataset import SegmentationDataset
import time
import segmentation_models_pytorch as smp
from config import *
from models.functions import training_loop, validation_loop, test_loop
from utils import utils
from utils.csv_file import CSV

output_dir="unet_model/"

# If folder doesn't exist, then create it.
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

train_values=CSV(output_dir+"train.csv", ['epoch','batch', 'loss', 'iou', 'f1', 'f2', 'accuracy', 'recall'])
class_train_values=CSV(output_dir+"train_class.csv", ['epoch', 'batch']+(list(ID_TO_NAME.values())))

validation_values=CSV(output_dir+"validation.csv", ['epoch','batch', 'loss', 'iou', 'f1', 'f2', 'accuracy', 'recall'])
class_val_values=CSV(output_dir+"validation_class.csv", ['epoch', 'batch']+(list(ID_TO_NAME.values())))



#Zbadać kwestię interpolacji (upewnić się, że mamy maski binarne po transformacji
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                                 ])

train_dataset = SegmentationDataset(IMAGE_TRAIN_PATH, transforms)
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

validation_dataset = SegmentationDataset(IMAGE_VALIDATION_PATH, transforms)
validation_loader = DataLoader(validation_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)


test_dataset = SegmentationDataset(IMAGE_TEST_PATH, transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=NUM_CLASSES,  # model output channels (number of classes in your dataset)
)

DEVICE = utils.get_device()

if torch.cuda.is_available():
    model.cuda()


lossFunc = CrossEntropyLoss()
opt = Adam(model.parameters(), lr=LEARNING_RATE)
print("[INFO] training UNET...")

train_loss=[]
val_loss=[]
startTime = time.time()
total_class_lossess = []
total_val_class_lossess = []

for e in tqdm(range(EPOCHS)):
    training_loop(model, opt, lossFunc, train_loader, DEVICE, e, train_values, class_train_values, total_class_lossess, train_loss)
    validation_loop(model, validation_loader, lossFunc, DEVICE, e, validation_values, class_val_values, total_val_class_lossess, val_loss)
    test_loop(model, test_loader, DEVICE, output_dir, e, "unet")
    utils.generate_train_val_plot(output_dir+"plot.png", train_loss, val_loss)
    utils.generate_class_loss_plot(output_dir+"class_plot.png", total_class_lossess)
    utils.generate_class_loss_plot(output_dir+"class_plot_val.png", total_val_class_lossess)