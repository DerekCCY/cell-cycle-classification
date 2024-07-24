import sys
sys.path.append('/home/ccy/cellcycle')
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from PIL import Image
from transform_alb import *
from transform import *
import torch.nn.functional as F


def prepare_plot(origImage, origMask, prediction, labelPath):
	
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	
	ax[0].imshow(origImage, cmap="gray")
	ax[1].imshow(origMask)
	ax[2].imshow(prediction)
	
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	
	filename = str("C_"+sorted(os.listdir(labelPath))[i])
	save_path = os.path.join(save_file,filename)
	figure.tight_layout()
	figure.show()
	figure.savefig(save_path)


color_to_label ={
    (0,0,0): 0,
    (0,0,255):1,    #rfp
    (0,255,0):2,    #gfp
    (0,255,255):3   #two phase
}
label_to_color = {v: k for k, v in color_to_label.items()}

def make_predictions(model, imagePath, labelPath_pic ,labelPath, savePath):
    	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		test_image_origin = cv2.imread(imagePath, cv2.IMREAD_COLOR)
  
		test_image = val_transform(image=test_image_origin)
		test_image = test_image['image']
		test_image = test_image.unsqueeze(0).float().cuda()
		print(type(test_image))
		print(test_image.shape)

		groundTruthPath_pic = os.path.join(labelPath_pic)
		test_label_pic = cv2.imread(groundTruthPath_pic, cv2.IMREAD_COLOR)
  
		groundTruthPath = os.path.join(labelPath)
		test_label = np.load(groundTruthPath)
		print(test_label.shape)
  
		prediction = model(test_image) #torch.Size([1, 4, 1024, 1024])
		print(prediction.shape)
		pred_prob = F.softmax(prediction, dim=1)
		pred_np = torch.argmax(pred_prob, dim=1).detach().cpu().numpy()
		print(pred_np.shape)
		
		predicted_label_image = np.zeros((pred_np.shape[1], pred_np.shape[2], 3), dtype=np.uint8)
		for i in range(predicted_label_image.shape[0]):
			for j in range(predicted_label_image.shape[1]):
				predicted_label_image[i,j] = label_to_color[pred_np[0,i,j]]
		print(predicted_label_image.shape)
		print(np.unique(predicted_label_image))

		''' Plot '''
		figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
		ax[0].imshow(test_image_origin)
		ax[1].imshow(test_label_pic)
		ax[2].imshow(predicted_label_image)
		ax[0].set_title("Image")
		ax[1].set_title("Original Mask")
		ax[2].set_title("Predicted Mask")
		figure.tight_layout()
		figure.show()
		figure.savefig(savePath)

print("[INFO] loading up test test_image paths...")
image_list = "/home/ccy/cellcycle/data/FUCCI/Split/test/bf_mito(REAL)_nuclei"
label_list_pic = "/home/ccy/cellcycle/data/FUCCI/Split/test/label"
label_list = "/home/ccy/cellcycle/data/FUCCI/Split/test/label_npy"
save_file = config.BASE_OUTPUT

for i in range(len(os.listdir(image_list))):
    imagePaths = os.path.join(image_list, sorted(os.listdir(image_list))[i])
    labelPaths_pic = os.path.join(label_list_pic, sorted(os.listdir(label_list_pic))[i])
    labelPaths= os.path.join(label_list, sorted(os.listdir(label_list))[i])
    savePaths = os.path.join(save_file, sorted(os.listdir(label_list_pic))[i])
    model = torch.load(config.MODEL_PATH).to(config.DEVICE)
    make_predictions(model, imagePath=imagePaths, labelPath_pic=labelPaths_pic, labelPath=labelPaths, savePath=savePaths)
