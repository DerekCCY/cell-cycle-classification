from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transform import *
from transform_alb import *
import cv2
import os 
from PIL import Image
import numpy as np

''' Dataset for mitochondria and nucleus prediction'''

class M_N_dataset(Dataset):
    
    def __init__(self, image_dir, label_dir, transform=None):
        # === Initialize paths, transforms, and so on === #
        self.images = sorted(os.listdir(str(image_dir)))
        self.labels = sorted(os.listdir(str(label_dir)))
        assert len(self.images) == len(self.labels)
        self.transform =transform 
        
        self.images_and_label = []
        for i in range(len(self.images)):
            self.images_and_label.append((str(image_dir)+'/'+str(self.images[i]),
                                          str(label_dir)+'/'+str(self.labels[i])))
    def __getitem__(self, index):
        # === Using Pillow === #
             
        image_path, label_path = self.images_and_label[index] # Read image path
        # === Image === #
        image = Image.open(image_path).convert("L") # PIL: RGB, OpenCV: BGR
        image = np.array(image)  # image -> numpy
        image = image.astype(np.uint8)  # uint8
        image = Image.fromarray(image)
        # === Label === #
        label = Image.open(label_path).convert("L")
        label = np.array(label)  
        label =  label.astype(np.uint8)
        label = Image.fromarray(label)
        # === Transform === #
        if self.transform is not None:
            image, label = self.transform((image,label))
        return image, label
       
    def __len__(self):
        return len(self.images)

''' Dataset for cell cycle prediction'''

class FucciDataset(Dataset):
    
    def __init__(self, image_dir, label_dir,transform=None):
        # === Initialize paths, transforms, and so on === #
        self.images = sorted(os.listdir(str(image_dir)))
        self.labels = sorted(os.listdir(str(label_dir)))
        assert len(self.images) == len(self.labels)
        self.transform =transform 
        
        self.images_and_label = []
        for i in range(len(self.images)):
            self.images_and_label.append((str(image_dir)+'/'+str(self.images[i]),
                                          str(label_dir)+'/'+str(self.labels[i])))
    def __getitem__(self, index):
        # === Using OpenCV === #
        
        image_path, label_path = self.images_and_label[index]  # Read image path
        # === Image === #
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
        image = image.astype(np.uint8)
        cv2.imwrite("image.png", image)
        ''' Label: npy'''
        # === Label === #
        label = np.load(label_path)
        
        ''' Label: image '
        # === Label === #
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.IMREAD_COLOR)
        label = label.astype(np.uint8)
        cv2.imwrite("label.png", label)
        '''
        # === Transform === #
        if self.transform is not None:
            transformed_image = self.transform(image=image, is_check_shapes=False)
            image = transformed_image['image']
            #label = transformed_image['mask']
        return image, label
    
    def __len__(self):
        return len(self.images)

''' Preprocessing for label '''

def label_preprocess(label_path):
    
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    # === Median Blur === #
    label = cv2.medianBlur(label,5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    #label = cv2.erode(label,kernel,iteration=1)
    # === Threshold === #
    threshold_value = 20  # You can adjust this threshold value as needed
    _, label = cv2.threshold(label, threshold_value, 255, cv2.THRESH_BINARY)
    # === Dilate === #
    label = cv2.dilate(label,kernel,iterations=8)
    
    label = label.astype(np.uint8)
    cv2.imwrite("label.png", label)