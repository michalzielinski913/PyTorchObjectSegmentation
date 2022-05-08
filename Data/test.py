import pickle
import numpy as np
from config import MASK_PATH
from utils.utils import visualize

with open(MASK_PATH +"8201.png", "rb") as f_in:
    mask = pickle.load(f_in)
    mask=np.where(mask > 2, 1, 0)
    mask = np.array(mask, dtype=np.uint8)
    print(mask[:, :, 0].shape)
    visualize("test.jpg", mask=mask[:, :, 0])