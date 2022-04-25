

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
EPOCHS=5
LEARNING_RATE = 0.001
INPUT_IMAGE_HEIGHT=256
INPUT_IMAGE_WIDTH=256
NUM_CLASSES=11