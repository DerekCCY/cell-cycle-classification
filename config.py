import os
import torch

''' Check cuda'''
DEVICE = "cuda" # determine the device to be used for training and evaluation
PIN_MEMORY = True if DEVICE == "cuda" else False # determine if we will be pinning memory during data loadingy

''' Parameters adjustment '''
INIT_LR = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 4

''' Image size '''
INPUT_IMAGE_WIDTH = 1024
INPUT_IMAGE_LENGTH = 1024

''' Directory for saving '''
BASE_MODEL = "/home/ccy/cellcycle/gfp_rfp_prediction/save_model"
MODEL_PATH = os.path.join(BASE_MODEL, "model_path/EUNet.pth")
CHECKPOINT_PATHS = os.path.join(BASE_MODEL, "model_path/EUnet_checkpoint.ckpt")
RECORD_PATH = os.path.join(BASE_MODEL,"record/EUNet_record2.csv")

BASE_OUTPUT = "/home/ccy/cellcycle/gfp_rfp_prediction/output"
PLOT_PATH = os.path.join(BASE_OUTPUT, "process/processEUNet.png")
PREDICT_PATHS = os.path.join(BASE_OUTPUT, "prediction")



