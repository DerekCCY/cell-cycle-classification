import torch
from torchvision import transforms
import numpy as np
import torchvision.transforms as transforms
import config
from PIL import ImageFilter

''' Standardization function '''

def max_min_normalize_image(image):
    min_value = np.min(image)
    max_value = np.max(image)
    normalized_image = (image - min_value) / (max_value - min_value)
    return normalized_image

def zero_normalize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    normalized_image = (image - mean) / std
    return normalized_image

class RandomTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x, y = sample
        seed = torch.randint(0, 2**32, size=(1,)).item()
        torch.manual_seed(seed)
        x = self.transform(x)
        torch.manual_seed(seed)
        y = self.transform(y)
        return x, y

        #image, label = sample
        #seed = torch.randint(0, 2**32, size=(1,)).item()
        #torch.manual_seed(seed)
        #transformed = self.transform(image=image, mask=label)
        #image = transformed["image"]
        #label = transformed["mask"]
        #return image, label

''' Training and validation transformation '''
train_transform = transforms.Compose(
    [   
        transforms.Resize((config.INPUT_IMAGE_LENGTH, config.INPUT_IMAGE_WIDTH)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomCrop(512, padding=5, pad_if_needed=False, padding_mode='edge'),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [   
        transforms.Resize((config.INPUT_IMAGE_LENGTH, config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor(),
    ]
)