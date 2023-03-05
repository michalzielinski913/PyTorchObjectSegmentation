# Pytorch Object Segmentation
Final project completed at the end of computer science course at Silesian University of Technology. It explores the application of semantic segmentation on aerial photos.

## Dataset
Dataset used for the training process was created by Silesian University of Technology students as a part of their curriculum. It contains about ~360 labelled aerial images with 11 classes in total.
![plot](./Assets/example_image.png) 
Currently there are no plans to release the dataset to the public.

## Data preprocessing
All scripts related to data preprocessing before training process are stored inside *Data/* directory.
<ol>
  <li>generate_masks.py: generate image files representing labels</li>
  <li>combine.py: combine all masks from specific image into single file</li>
  <li>dataset_split.py: Splitting data into train/validation/test set</li>
  <li>Fourth item</li>
</ol>

## Training
This project tests 4 different networks: U-Net, U-Net++, DeepLabV3 and DeepLabV3+. Entire training flow is controlled by *full_iteration.py* script inside *models/* directory. Each epoch will generate results inside csv files and some basic plots for visualization.

## Detection
Detection can be performed by *eval.py* script inside main project directory with the following syntax:
```
usage: python eval.py [options]
options:
-d <argument>, --directory <argument>       Path to the folder where images are stored
-f <argument>, --file <argument>            Path to the file
User must provide one of two options mentioned above
-a <argument>, --architecture <argument>    Name of used architecture, possible choices: unet, unet++, deeplab and deeplab+
-e <argument>, --encoder <argument>         Name of the encoder, possible choices: resnet50, resnext50 and efficientnet
-s <argument>, --size <argument>            Size of image during detection, possible choices: 512, 768 and 1024
-w <argument>, --weights <argument>         Path to the zip file containing model weights. They must match model defined using other parameters
-o <argument>, --output <argument>          Path to the folder where results will be stored, if not provided script location will be used
Please note that all predictions will be saved in given output directory as prediction_[original file name]
ex. python eval.py -a unet -e efficientnet -s 1024 -w model.zip -f demo.jpg -o img/

```

## Languages Used

This application was created using Python with libraries such as PyTorch, Segmentation Models, Pandas and Sci-Kit.

## License

This application is licensed under the MIT License. See the LICENSE file for more information.
