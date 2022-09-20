"""
This file is responsible for generating all plots
"""
from utils.utils import generate_plot

metrics = ['iou', 'f1', 'f2', 'accuracy', 'recall']
dirs = ['deeplabv3_model/', 'deeplabv3plus_model/', 'unet_model/']

for model in dirs:
    print(model)
    for metric in metrics:
        generate_plot(train_csv_path=model + "train.csv", val_csv_path=model + "validation.csv", metric=metric, save_location=model + metric + ".jpg")
