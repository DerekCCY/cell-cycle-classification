import numpy as np
import cv2
import os
import torch.nn.functional as F
import torch

def find_closest_color(pixel, color_dict):
    
    closest_color = None
    min_distance = float('inf')
    
    for color, label in color_dict.items():
        
        distance = np.linalg.norm(np.array(pixel) - np.array(color))
        #print(distance)
        if distance < min_distance:
            min_distance = distance
            closest_color = label
            
    #print(f"closet_color {closest_color}")
    return closest_color

def label_image_to_class(parent_path, file, save_dir):
    
    color_to_label ={
        (0,0,0): 0,
        (0,0,255):1,    #rfp
        (0,255,0):2,    #gfp
        (0,255,255):3   #two phase
    }
    
    #print(color_to_label)
    for label_name in range(len(file)):
        
        label_image = cv2.imread(os.path.join(parent_path, file[label_name]))
        label_image = cv2.resize(label_image, (1024, 1024))
        label_array = np.zeros((label_image.shape[0], label_image.shape[1]), dtype=np.uint8)
        
        for i in range(label_image.shape[0]):
            for j in range(label_image.shape[1]):
                
                pixel = tuple(label_image[i,j])
                #print(f'pixel:{pixel}')
                if pixel in color_to_label:
                    label_array[i, j] = color_to_label[pixel]
                    #print(label_array[i,j])
                else:    
                    label_array[i, j] = find_closest_color(pixel, color_to_label)
                    #print(label_array[i,j])
                    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        #print(label_array)
        #print(label_array.shape)
        label_tensor = torch.tensor(label_array, dtype=torch.long) # Convert label_array to a PyTorch Tensor
        one_hot_labels = F.one_hot(label_tensor, num_classes=4) # One-hot encode the file tensor
        #print(one_hot_labels.numpy().shape)
        print(np.unique(one_hot_labels.numpy()))
        
        npy_data_name = file[label_name].replace('.png','.npy')
        np.save(os.path.join(save_dir,npy_data_name), one_hot_labels.numpy())
    
def main():
    label_image_path = '/home/ccy/cellcycle/data/FUCCI/label'
    sorted_label_image = sorted(os.listdir(label_image_path))
    save_dir = '/home/ccy/cellcycle/data/FUCCI/label_npy'
    label_image_to_class(parent_path=label_image_path, file=sorted_label_image, save_dir=save_dir)
                    
if __name__ == '__main__':
    main()          