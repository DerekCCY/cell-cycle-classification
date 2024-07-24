import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

''' torch '''

def f_score(y_true, y_pred, beta=1, eps=1e-7, threshold=None):
    
    if threshold is not None:
        y_pred = (y_pred > threshold).float()

    tp = torch.sum(y_true * y_pred)
    fp = torch.sum(y_pred) - tp
    fn = torch.sum(y_true) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
    return f_score

''' Numpy '''

def f_score(y_true, y_pred, beta=1, eps=1e-7, threshold=None):
    
    if threshold is not None:
        y_pred = (y_pred > threshold).astype(np.float32)

    tp = np.sum(y_true * y_pred)
    fp = np.sum(y_pred) - tp
    fn = np.sum(y_true) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
    return f_score

def calculate_iou(pred, target, threshold=0.5):
    
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum().item()
    union = ((pred_binary + target_binary) > 0).sum().item()

    iou = intersection / (union + 1e-7) 
    
    return iou


# Apply softmax to get probabilities

def calculate_accuracy(pred, y):
    
    pred_np = pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    pred_np = np.argmax(pred_np,axis=3)
    y_np = np.argmax(y_np,axis=3)
    
    # Debug print statements
    #print(f"pred_np: {pred_np.shape}")
    #print(f"y_np: {y_np.shape}")
    
    # Flatten the arrays
    pred_np = pred_np.flatten()
    y_np = y_np.flatten()
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_np, pred_np, average="weighted")
    
    return f1