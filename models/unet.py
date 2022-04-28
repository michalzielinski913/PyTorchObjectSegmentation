import torch
import os
import csv

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import utils.utils
from Dataset import SegmentationDataset
import time
import segmentation_models_pytorch as smp
from utils import *
from config import *

###TODO: Rozwiązać kwestię zdjęć testowych
### Notatka, hardcode listę plików aby każdy model miał taki sam dataset

output_dir="unet_model/"

# If folder doesn't exist, then create it.
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


train_csv_file=open(output_dir+"train.csv", mode='w', newline='')
train_writer = csv.writer(train_csv_file, delimiter=';')
train_writer.writerow(['epoch','batch', 'loss'])

validation_csv_file=open(output_dir+"validation.csv", mode='w', newline='')
validation_writer = csv.writer(validation_csv_file, delimiter=';')
validation_writer.writerow(['epoch','batch', 'loss'])

files = os.listdir(IMAGE_PATH)

#Zbadać kwestię interpolacji (upewnić się, że mamy maski binarne po transformacji
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
                                 ])
#Pociąć maski zamiast resize, przetestować
train_size = int(TRAIN_RATIO * len(files))

train_dataset = SegmentationDataset(IMAGE_PATH, MASK_PATH, files[0:20], transforms)
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

validation_dataset = SegmentationDataset(IMAGE_PATH, MASK_PATH, files[21:26], transforms)
validation_loader = DataLoader(validation_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=NUM_CLASSES,  # model output channels (number of classes in your dataset)
)

DEVICE = utils.get_device()

if torch.cuda.is_available():
    model.cuda()
lossFunc = smp.losses.JaccardLoss(mode='multilabel')
opt = Adam(model.parameters(), lr=LEARNING_RATE)
print("[INFO] training UNET...")

train_loss=[]
val_loss=[]

startTime = time.time()
for e in tqdm(range(EPOCHS)):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalTestLoss = 0
    # loop over the training set
    for (i, (x, y)) in enumerate(train_loader):
        # send the input to the device
        (x, y) = (x.to(DEVICE), y.to(DEVICE))

        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFunc(pred, y)
        print("[Train] {}/{}, Loss:{:.3f}".format(i, len(train_loader), loss/TRAIN_BATCH_SIZE))
        opt.zero_grad()
        loss.backward()
        opt.step()
        totalTrainLoss += loss.cpu().detach().item()
        train_writer.writerow([e, i, loss.cpu().detach().item()])
    epoch_train_loss=totalTrainLoss / (int(len(train_dataset)))
    train_loss.append(epoch_train_loss)
    print("Train loss: {:.6f}".format(epoch_train_loss))
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (i, (x, y)) in enumerate(validation_loader):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # make the predictions and calculate the validation loss
            pred = model(x)
            loss=lossFunc(pred, y).cpu().detach().item()
            validation_writer.writerow([e, i, loss])
            totalTestLoss += loss
            print("[Validation] {}/{}, Loss:{:.3f}".format(i, len(validation_loader), loss/VAL_BATCH_SIZE))
        epoch_val_loss=totalTestLoss / (int(len(validation_dataset)))
        val_loss.append(epoch_val_loss)
        print("Test loss avg: {:0.6f}".format(epoch_val_loss))

        epoch_dir=output_dir+"/epoch_"+str(e)
        if not os.path.isdir(epoch_dir):
            os.makedirs(epoch_dir)

        for (i, (x, y)) in enumerate(validation_loader):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # make the predictions and calculate the validation loss
            pred = model(x)
            for label in range(len(pred[0])):
                filename="{}/{}_{}.png".format(epoch_dir, i, label)
                utils.visualize(filename, Image=x[0].cpu().data.numpy(), Prediction=pred.cpu().data.numpy()[0][label], RealMask=y.cpu().data.numpy()[0][label])
    torch.save(model.state_dict(), os.path.join(epoch_dir+"/", 'unet_' + str(e) + '.zip'))
utils.generate_train_val_plot(output_dir, train_loss, val_loss)