import sys
sys.path.append('/home/ccy/cellcycle')

import argparse
import cv2
from config import *
import os
from tqdm import tqdm

from dataset import FucciDataset
from transform_alb import train_transform, val_transform
from torch.utils.data import DataLoader

from model.UNet import Modified_UNet, UNet
from model.UNet3Plus import UNet_3Plus
from model.EUNet import EffUNet_b0

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from utils.record import *
from utils.score import *
from utils.loss_function import *
from utils.check_function import *
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

''' Check cuda'''
#check_cuda()

''' Image Path '''
train_img_path = 'data/FUCCI/Split/train/bf_mito(REAL)_nuclei'
train_label_path = 'data/FUCCI/Split/train/label_npy'
val_img_path = 'data/FUCCI/Split/val/bf_mito(REAL)_nuclei'
val_label_path = 'data/FUCCI/Split/val/label_npy'

''' Data Ready '''
trainDS = FucciDataset(image_dir=train_img_path, label_dir=train_label_path, transform=train_transform)
valDS =  FucciDataset(image_dir=val_img_path, label_dir=val_label_path, transform=val_transform)
print(f"[INFO] {len(trainDS)} examples in the training set...")
print(f"[INFO] {len(valDS)} examples in the val set...")
trainSteps = len(trainDS) // config.BATCH_SIZE
valSteps = len(valDS) // config.BATCH_SIZE

trainLoader = DataLoader(trainDS, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
valLoader = DataLoader(valDS, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)

''' Check Image'''
#check_input_image(trainLoader)

''' Model Selection '''
net = Modified_UNet()
unet= UNet(3,1)
unet3plus = UNet_3Plus(3,1)
eunet = EffUNet_b0(3,4)

''' Loss function Selection '''
focaldice_loss = FocalDiceLoss(alpha=0.5, gamma=1, smooth=1e-5)

def parse_args():
    parser = argparse.ArgumentParser(description="Training a model for cell cycle classification")
    #parser.add_argument("--model_type", type=str, default="eunet")
    #parser.add_argument("--loss_type", type=str, default="focaldice_loss")
    parser.add_argument("--lr", type=float, default=config.INIT_LR)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    return parser.parse_args()

''' Training function '''
def train(model, trainLoader, optimizer, gradient, loss_fn ,device):
    model.train()
    total_loss, total_f1 = 0, 0
    for batch_idx, (x, y) in enumerate(trainLoader):
        x, y = x.float().to(device), y.to(device)
        pred = model(x)
        pred = pred.permute(0,2,3,1)
        #print(f"pred: {pred.shape}")   [4, 1024,1024,4]
        #print(f"y: {y.shape}")  # [4, 1024, 1024, 4]
        
        # === Loss === #
        loss = loss_fn(pred, y)
        loss.backward() 
        total_loss += loss.item() # prevent tensor accumulating infinitely
        
        
        if (batch_idx + 1) % gradient == 0:
            optimizer.zero_grad()      # Zero gradients
            optimizer.step()
        
        # === Accuracy === #
        f1 = calculate_accuracy(pred,y)
        total_f1 += f1
        #confu_metric = confusion_matrix(y_np, pred_np)
        #total_pcc += pearsonr(pred_np, y_np)[0]
        
    return total_loss / len(trainLoader), total_f1 / len(trainLoader)

''' Validation function'''
def validate(model, valLoader, loss_fn, device):
    model.eval()
    total_loss, total_f1 = 0, 0
    with torch.no_grad():
        for x, y in valLoader:
            x, y = x.float().to(device), y.to(device)
            pred = model(x)
            pred = pred.permute(0,2,3,1)
            
            # === Loss === #
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            
            # === Accuracy === #
            f1 = calculate_accuracy(pred,y)
            total_f1 += f1
            #confu_metric = confusion_matrix(y_np, pred_np)
            #total_pcc += pearsonr(pred_np, y_np)[0]
            
    return total_loss / len(valLoader), total_f1 / len(valLoader)

''' ======================================================================================== '''

def main():
    args = parse_args()

    # === Model selection === #
    model = eunet.to(config.DEVICE)
    
    # ==== Loss function selection === #
    loss_fn = focaldice_loss

    # === Optimizer === #
    opt = Adam(model.parameters(), lr=config.INIT_LR,betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(opt,mode='min', factor=0.1, patience=10, verbose=True, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    # === Parameters Initialization === #
    epoch_train_losses, epoch_valid_losses = [], []
    epoch_train_f1s, epoch_valid_f1s = [], []
    stale = 0
    best_acc = 0.0
    
    # === Statistics Loading === #
    stats = TrainingStatistics(config.RECORD_PATH)
    if os.path.exists(config.RECORD_PATH):
        loaded_values = stats.load()
        epoch_train_losses, epoch_valid_losses, epoch_train_f1s, epoch_valid_f1s = map(list, zip(*loaded_values))
    if os.path.exists(config.CHECKPOINT_PATHS):
        modelstats = ModelStats
        _, model, _, best_acc = modelstats.load_model(path=config.CHECKPOINT_PATHS, epoch=0, model=model, optimizer=opt, best_acc=best_acc, device= config.DEVICE)
    
    # === Training === #
    for epoch in tqdm(range(args.epochs)):
        train_loss, train_f1 = train(model=model, trainLoader=trainLoader, optimizer=opt, gradient=4, loss_fn=loss_fn, device=config.DEVICE)
        valid_loss, valid_f1 = validate(model=model, valLoader=valLoader, loss_fn=loss_fn, device=config.DEVICE)
        scheduler.step(valid_loss)

        if valid_f1 > best_acc:
            stale = 0
            best_acc = valid_f1
            ModelStats.save_checkpoint(epoch+1, model, opt, best_acc, config.CHECKPOINT_PATHS)
            torch.save(model, config.MODEL_PATH)
        else: 
            stale += 1
            if stale == 20:
                print(f'No improvment 20 consecutive epochs, early stopping')
                break
        
        epoch_train_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)
        epoch_train_f1s.append(train_f1)
        epoch_valid_f1s.append(valid_f1)    
        stats.save(epoch_train_losses, epoch_valid_losses, epoch_train_f1s, epoch_valid_f1s)
        
        '''Print in terminal'''
        tqdm.write(f"Epoch: {epoch}/{100}")
        tqdm.write(f"trainingLoss:{train_loss}  validationLoss:{valid_loss}")
        tqdm.write(f"trainingAccuracy:{train_f1}  validationAccuracy:{valid_f1}")
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        tqdm.write("Learning rate: {}".format(current_lr))
         
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Train PCC: {train_f1}, Valid PCC: {valid_f1}, Learning Rate: {current_lr}")

if __name__ == "__main__":
    main()

