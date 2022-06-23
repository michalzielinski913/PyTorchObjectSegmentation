import torch
import os
import csv

import torchvision
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import utils.utils
from Data.Dataset import SegmentationDataset
import time
import segmentation_models_pytorch as smp
from config import *
from utils import *
from utils.csv_file import CSV

output_dir="unet_model/"

# If folder doesn't exist, then create it.
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

train_values=CSV(output_dir+"train.csv", ['epoch','batch', 'loss'])
class_train_values=CSV(output_dir+"train_class.csv", ['epoch', 'batch']+(list(ID_TO_NAME.values())))

validation_values=CSV(output_dir+"validation.csv", ['epoch','batch', 'loss'])
class_val_values=CSV(output_dir+"validation_class.csv", ['epoch', 'batch']+(list(ID_TO_NAME.values())))

files = os.listdir(IMAGE_PATH)
files=[item for item in files if item not in TEST_IMAGES_FILENAMES]

#Zbadać kwestię interpolacji (upewnić się, że mamy maski binarne po transformacji
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                                 ])
#Pociąć maski zamiast resize, przetestować
train_size = int(TRAIN_RATIO * len(files))

train_dataset = SegmentationDataset(IMAGE_PATH, MASK_PATH, files[0:train_size], transforms)
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

validation_dataset = SegmentationDataset(IMAGE_PATH, MASK_PATH, files[train_size:], transforms)
validation_loader = DataLoader(validation_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

test_dataset = SegmentationDataset(IMAGE_PATH, MASK_PATH, TEST_IMAGES_FILENAMES, transforms)
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
class_weights = torch.ones([10])
class_weights = torch.reshape(class_weights,(1,10,1,1)).to(device="cuda")
lossFunc = BCEWithLogitsLoss(pos_weight=class_weights)
lossFunc_two=BCEWithLogitsLoss()
opt = Adam(model.parameters(), lr=LEARNING_RATE)
print("[INFO] training UNET...")

train_loss=[]
val_loss=[]
startTime = time.time()
total_class_lossess = []
total_val_class_lossess = []

for e in tqdm(range(EPOCHS)):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalTestLoss = 0
    class_losses=[0]*NUM_CLASSES
    val_class_losses=[0]*NUM_CLASSES

    # loop over the training set
    for (i, (x, y)) in enumerate(train_loader):
        # send the input to the device
        (x, y) = (x.to(DEVICE), y.to(DEVICE))

        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFunc(pred, y)
        class_losses_batch = [0] * NUM_CLASSES
        for class_id in range(NUM_CLASSES):
            class_losses_batch[class_id]=lossFunc_two(pred[:,class_id],y[:,class_id]).cpu().detach().item()
            class_losses[class_id]+=lossFunc_two(pred[:,class_id],y[:,class_id]).cpu().detach().item()
        class_train_values.writerow([e, i]+(class_losses_batch))
        print("[Train] {}/{}, Loss:{:.3f}".format(i, len(train_loader), loss))
        opt.zero_grad()
        loss.backward()
        opt.step()
        totalTrainLoss += loss.cpu().detach().item()
        train_values.writerow([e, i, loss.cpu().detach().item()])
    class_losses = [number / (int(len(train_dataset)/TRAIN_BATCH_SIZE)) for number in class_losses]
    total_class_lossess.append(class_losses)
    epoch_train_loss=totalTrainLoss / (int(len(train_dataset)/TRAIN_BATCH_SIZE))
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
            class_losses_batch = [0] * NUM_CLASSES
            for class_id in range(NUM_CLASSES):
                class_losses_batch[class_id] = lossFunc_two(pred[:, class_id], y[:, class_id]).cpu().detach().item()
                val_class_losses[class_id] += lossFunc_two(pred[:, class_id], y[:, class_id]).cpu().detach().item()
            validation_values.writerow([e, i, loss])
            class_val_values.writerow([e, i]+(class_losses_batch))

            totalTestLoss += loss
            print("[Validation] {}/{}, Loss:{:.3f}".format(i, len(validation_loader), loss))
        epoch_val_loss=totalTestLoss / (int(len(validation_dataset)/VAL_BATCH_SIZE))
        val_class_losses = [number / (int(len(validation_dataset) / VAL_BATCH_SIZE)) for number in val_class_losses]
        total_val_class_lossess.append(val_class_losses)
        val_loss.append(epoch_val_loss)
        print("Test loss avg: {:0.6f}".format(epoch_val_loss))

        epoch_dir=output_dir+"/epoch_"+str(e)
        if not os.path.isdir(epoch_dir):
            os.makedirs(epoch_dir)

        for (i, (x, y)) in enumerate(test_loader):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # make the predictions and calculate the validation loss
            pred = model(x)
            pred = torch.sigmoid(pred)
            for label in range(len(pred[0])):
                filename="{}/{}_{}.png".format(epoch_dir, i, label)
                utils.visualize(filename, label, Image=x[0].cpu().data.numpy(), Prediction=pred.cpu().data.numpy()[0][label].round(), RealMask=y.cpu().data.numpy()[0][label])
            utils.confusion_matrix("{}/{}_matrix.png".format(epoch_dir, i),pred, y)
    torch.save(model.state_dict(), os.path.join(epoch_dir+"/", 'unet_' + str(e) + '.zip'))
    utils.generate_train_val_plot(output_dir+"plot.png", train_loss, val_loss)
    utils.generate_class_loss_plot(output_dir+"class_plot.png", total_class_lossess)
    utils.generate_class_loss_plot(output_dir+"class_plot_val.png", total_val_class_lossess)