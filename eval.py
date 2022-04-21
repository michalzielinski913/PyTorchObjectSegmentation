import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchvision import transforms
import numpy as np
from Dataset import SegmentationDataset
import cv2
import matplotlib.pyplot as plt

MODEL_PATH="checkpoint/checkpoint_sm_15.zip"
IMAGE_PATH="F:\\Poles\\Dataset\\Image\\"
MASK_PATH="F:\\Poles\\Dataset\\Combine\\"

IMAGES=["5.jpg"]
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
INPUT_IMAGE_HEIGHT=512
INPUT_IMAGE_WIDTH=512
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((INPUT_IMAGE_HEIGHT,INPUT_IMAGE_WIDTH)),
	                            transforms.ToTensor()])

eval_dataset=SegmentationDataset(IMAGE_PATH, MASK_PATH, IMAGES, transform)
evalLoader=DataLoader(eval_dataset, batch_size=1, shuffle=True)

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print('Running on the GPU')
else:
    DEVICE = "cpu"
    print('Running on the CPU')

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)
if torch.cuda.is_available():
	model.cuda()
model.load_state_dict(torch.load(MODEL_PATH))
mask=None
x,y=eval_dataset[0]
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()
    for (i, (x, y)) in enumerate(evalLoader):
        # send the input to the device
        (x, y) = (x.to(DEVICE), y.to(DEVICE))
        # perform a forward pass and calculate the training loss
        pred = model(x)
        pred=pred.sigmoid()
        mask=pred.cpu().data.numpy()
print(mask.shape)
visualize(mask=mask[0][0])
visualize(mask=mask[0][1])
visualize(mask=mask[0][2])
unique, counts = np.unique(mask[0][1], return_counts=True)
print(list(zip(unique, counts)))
mask[mask>=0.1]=255
mask[mask<0.1]=0


visualize(mask=mask[0][0])
visualize(mask=mask[0][1])
visualize(mask=mask[0][2])
cv2.imwrite("test1.png", mask[0][0])
cv2.imwrite("test2.png", mask[0][1])
cv2.imwrite("test3.png", mask[0][2])
