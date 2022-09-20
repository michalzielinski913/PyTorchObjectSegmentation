import os

import torch
from torch.utils.data import DataLoader

from config import NUM_CLASSES, TRAIN_BATCH_SIZE
from utils import utils
from utils.csv_file import CSV


def training_loop(model, optimizer, loss_function, loader: DataLoader, DEVICE: str, epoch_number: int,
                  train_values: CSV, class_train_values: CSV, total_class_lossess: list, train_loss: CSV):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    class_losses=[0]*NUM_CLASSES
    # loop over the training set
    for (i, (x, y)) in enumerate(loader):
        # send the input to the device
        (x, y) = (x.to(DEVICE), y.to(DEVICE))

        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = loss_function(pred, y)
        class_losses_batch = [0] * NUM_CLASSES
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for class_id in range(NUM_CLASSES):
            class_losses_batch[class_id]=loss_function(pred[:,class_id],y[:,class_id]).cpu().detach().item()
            class_losses[class_id]+=loss_function(pred[:,class_id],y[:,class_id]).cpu().detach().item()
        class_train_values.writerow([epoch_number, i]+(class_losses_batch))
        print("[Train] {}/{}, Loss:{:.3f}".format(i, len(loader), loss))
        totalTrainLoss += loss.cpu().detach().item()
        iou, f1, f2, accuracy, recall=utils.metrics_calculation(pred, y)
        train_values.writerow([epoch_number, i, loss.cpu().detach().item(), iou, f1, f2, accuracy, accuracy])
    class_losses = [number / (int(len(loader))) for number in class_losses]
    total_class_lossess.append(class_losses)
    epoch_train_loss=totalTrainLoss / (int(len(loader)))
    train_loss.append(epoch_train_loss)
    print("Train loss: {:.6f}".format(epoch_train_loss))

def validation_loop(model, loader, loss_function, DEVICE: str, epoch_number: int, validation_values, class_val_values,
                    total_val_class_lossess: list, val_loss: list):
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        totalTestLoss = 0
        val_class_losses = [0] * NUM_CLASSES

        # loop over the validation set
        for (i, (x, y)) in enumerate(loader):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # make the predictions and calculate the validation loss
            pred = model(x)
            loss=loss_function(pred, y).cpu().detach().item()
            class_losses_batch = [0] * NUM_CLASSES
            for class_id in range(NUM_CLASSES):
                class_losses_batch[class_id] = loss_function(pred[:, class_id], y[:, class_id]).cpu().detach().item()
                val_class_losses[class_id] += loss_function(pred[:, class_id], y[:, class_id]).cpu().detach().item()
            iou, f1, f2, accuracy, recall = utils.metrics_calculation(pred, y)

            validation_values.writerow([epoch_number, i, loss, iou, f1, f2, accuracy, recall])
            class_val_values.writerow([epoch_number, i]+(class_losses_batch))

            totalTestLoss += loss
            print("[Validation] {}/{}, Loss:{:.3f}".format(i, len(loader), loss))
        epoch_val_loss=totalTestLoss / (int(len(loader)))
        val_class_losses = [number / (int(len(loader))) for number in val_class_losses]
        total_val_class_lossess.append(val_class_losses)
        val_loss.append(epoch_val_loss)
        print("Validation loss avg: {:0.6f}".format(epoch_val_loss))

def test_loop(model, loader, DEVICE, output_dir, epoch_number, model_name):
    with torch.no_grad():
        epoch_dir = output_dir + "/epoch_" + str(epoch_number)
        if not os.path.isdir(epoch_dir):
            os.makedirs(epoch_dir)

        for (i, (x, y)) in enumerate(loader):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # make the predictions and calculate the validation loss
            pred = model(x)
            pred = torch.sigmoid(pred)
            filename = "{}/{}_predictions.png".format(epoch_dir, i)
            utils.visualize(filename, Image=x[0].cpu().data.numpy(),
                            Prediction=pred.cpu().data.numpy()[0].round(),
                            RealMask=y.cpu().data.numpy()[0])

            utils.confusion_matrix("{}/{}_matrix.png".format(epoch_dir, i), pred, y)
    torch.save(model.state_dict(), os.path.join(epoch_dir + "/", model_name+'_' + str(epoch_number) + '.zip'))
