

### Paths ###
IMAGE_TRAIN_PATH="G:\\Dataset\\Split\\Train\\split\\"

IMAGE_VALIDATION_PATH="G:\\Dataset\\Split\\Validation\\split\\"

IMAGE_TEST_PATH="G:\\Dataset\\Split\\Test\\split\\"

MODEL_PATH="checkpoint/checkpoint_sm_15.zip"
OUTPUT_PATH="output/"

### Dataset settings ###
TRAIN_RATIO=0.8
TEST_IMAGES=4
TRAIN_BATCH_SIZE=4
VAL_BATCH_SIZE=1

### Model settings ###
EPOCHS=50
LEARNING_RATE = 3e-4
INPUT_IMAGE_HEIGHT=512
INPUT_IMAGE_WIDTH=512
NUM_CLASSES=10


DETECTION_THRESHOLD=0.5
TEST_IMAGES_FILENAMES=["8200.jpg", "8535.jpg", "8465.jpg", "8459.jpg", "8438.jpg"]

ID_TO_NAME = {
    0: "Building",
    1: "Vechicle",
    2: "Tree's shadow",
    3: "Landfill",
    4: "Parking",
    5: "Heap of earth ",
    6: "Excavation",
    7: "WaterContainer",
    8: "FieldParking",
    9: "MultipleTrees",
}