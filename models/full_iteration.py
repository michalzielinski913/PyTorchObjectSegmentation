import torch
import os
import torchvision
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, LovaszLoss, SoftBCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from Data.Dataset import SegmentationDataset
import segmentation_models_pytorch as smp
from config import *
from models.functions import training_loop, validation_loop, test_loop
from utils import utils
from utils.communication import send_email
from utils.csv_file import CSV

### Model loop function ###
from utils.utils import generate_metric_plots


def train_model(model, path):
    """
    Train single model
    :param model: segmentation models network
    :param path: path where results will be stored  for example "deeplabv3plus_model/"
    """
    output_dir = path
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    train_values = CSV(output_dir + "train.csv", ['epoch', 'batch', 'loss', 'iou', 'f1', 'f2', 'accuracy', 'recall'])
    class_train_values = CSV(output_dir + "train_class.csv", ['epoch', 'batch'] + (list(ID_TO_NAME.values())))

    validation_values = CSV(output_dir + "validation.csv",
                            ['epoch', 'batch', 'loss', 'iou', 'f1', 'f2', 'accuracy', 'recall'])
    class_val_values = CSV(output_dir + "validation_class.csv", ['epoch', 'batch'] + (list(ID_TO_NAME.values())))
    DEVICE = utils.get_device()

    if torch.cuda.is_available():
        model.cuda()

    lossFunc = DiceLoss(mode="multilabel")
    opt = Adam(model.parameters(), lr=LEARNING_RATE)
    print("[INFO] training {}...".format(str(model.__class__.__name__)))

    train_loss = []
    val_loss = []
    total_class_lossess = []
    total_val_class_lossess = []

    for e in tqdm(range(EPOCHS)):
        training_loop(model, opt, lossFunc, train_loader, DEVICE, e, train_values, class_train_values,
                      total_class_lossess, train_loss)
        validation_loop(model, validation_loader, lossFunc, DEVICE, e, validation_values, class_val_values,
                        total_val_class_lossess, val_loss)
        test_loop(model, test_loader, DEVICE, output_dir, e, "model")
        utils.generate_train_val_plot(output_dir + "plot.png", train_loss, val_loss)
        utils.generate_class_loss_plot(output_dir + "class_plot.png", total_class_lossess)
        utils.generate_class_loss_plot(output_dir + "class_plot_val.png", total_val_class_lossess)
    generate_metric_plots(output_dir)

###########################


#### UNIVERSAL CODE #########
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                                 ])

train_dataset = SegmentationDataset(IMAGE_TRAIN_PATH, transforms)
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True)

validation_dataset = SegmentationDataset(IMAGE_VALIDATION_PATH, transforms)
validation_loader = DataLoader(validation_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, pin_memory=True)


test_dataset = SegmentationDataset(IMAGE_TEST_PATH, transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#####################################

try:
    ### DEEPLABV3 PLUS ##################
    output_dir="deeplabv3plus_model_dice/"
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=NUM_CLASSES,  # model output channels (number of classes in your dataset)
    )
    train_model(model, output_dir)
    #####################################
except Exception as e:
    send_email(content="DeepLabV3 Plus failed")
send_email(content="DeepLabV3 Plus finished")

try:
    ### DEEPLABV3 ##################
    output_dir="deeplabv3_model_dice/"
    model = smp.DeepLabV3(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=NUM_CLASSES,  # model output channels (number of classes in your dataset)
    )
    train_model(model, output_dir)
    #####################################
except Exception as e:
    send_email(content="DeepLabV3 failed")
send_email(content="DeepLabV3 finished")

try:
    ### UNET ##################
    output_dir="unet_model_dice/"
    model = smp.DeepLabV3(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=NUM_CLASSES,  # model output channels (number of classes in your dataset)
    )
    train_model(model, output_dir)
    #####################################
except Exception as e:
    send_email(content="UNET failed")
send_email(content="UNET failed")

send_email()