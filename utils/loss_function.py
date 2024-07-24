import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''Regression Loss'''
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        diff = torch.abs(input - target)
        loss = torch.where(diff < self.delta, 0.5 * diff**2, self.delta * (diff - 0.5 * self.delta))
        return loss.mean()

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, input, target):
        diff = input - target
        loss = torch.log(torch.cosh(diff))
        return loss.mean()
    
'''Classification Loss'''
class FocalBCELoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=None, reduction='mean'):
        super(FocalBCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_loss = nn.BCELoss(reduction='none')
        self.reduction = reduction
        
    def forward(self, input, target):
        bce_loss = self.bce_loss(input, target)
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class FocalDiceLoss(nn.Module):
    
    def __init__(self, alpha=0.5, gamma=1, smooth=1e-5):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
    def dice_loss(self, label, prediction):
        
        prediction = prediction.float()
        prediction = F.softmax(prediction,dim=3)
        label = F.softmax(label,dim=3)

        intersection = torch.sum(label*prediction, dim=(1,2))  # sum along height and width
        union = torch.sum(label, dim=(1,2))+torch.sum(prediction,dim=(1,2))
        
        dice = (2. * intersection + self.smooth) / (union+self.smooth)
        return 1 - torch.mean(dice) # mean across batch
    
    def focal_loss(self, label, prediction):
        
        prediction = prediction.float()
        prediction = F.softmax(prediction,dim=3)
        label = F.softmax(label,dim=3)
        
        BCE = -label * torch.log(prediction + self.smooth)
        focal_loss = self.alpha * (1 - prediction)**self.gamma * BCE
        return torch.mean(focal_loss)
    
    def forward(self, label, prediction):
        dice_loss = self.dice_loss(label, prediction)
        focal_loss = self.focal_loss(label, prediction)
        return dice_loss + focal_loss
