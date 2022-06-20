

### Paths ###
IMAGE_PATH="F:\\Poles\\Dataset\\Image2\\"
MASK_PATH="F:\\Poles\\Dataset\\Combine2\\"
MODEL_PATH="checkpoint/checkpoint_sm_15.zip"
OUTPUT_PATH="output/"

### Dataset settings ###
TRAIN_RATIO=0.8
TEST_IMAGES=4
TRAIN_BATCH_SIZE=4
VAL_BATCH_SIZE=1

### Model settings ###
EPOCHS=50
LEARNING_RATE = 0.001
INPUT_IMAGE_HEIGHT=512
INPUT_IMAGE_WIDTH=512
NUM_CLASSES=11


DETECTION_THRESHOLD=0.5
TEST_IMAGES_FILENAMES=["8200.jpg", "8535.jpg", "8465.jpg", "8459.jpg", "8438.jpg"]

ID_TO_NAME = {
    0: "Budynek",
    1: "Drzewo",
    2: "Pojazd",
    3: "WieleDrzew",
    4: "CienieDrzew",
    5: "Skladowisko",
    6: "Parking",
    7: "PryzmaZiemi",
    8: "Wykop",
    9: "ZbiornikWodny",
    10: "DzikiParking",
}