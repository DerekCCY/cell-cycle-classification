import torch
import cv2
import config

def check_cuda():
    print(f"torch.__version__{torch.__version__}")
    print(f'torch.cuda.is_available() {torch.cuda.is_available()}')
    print(f"torch.cuda.device_count {torch.cuda.device_count()}")
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)
    print(torch.cuda.current_device())
    
def check_input_image(dataloader):
    images, labels = next(iter(dataloader))
    for i in range(config.BATCH_SIZE):
        image1 = images[i] #(C, H, W)
        label1 = labels[i] #(C, H, W)
        image_np = image1.permute(1,2,0).numpy()
        label_np = label1.numpy()
        #print(image_np.shape)
        #print(label_np.shape)
        cv2.imwrite(f"/home/ccy/cellcycle/test/image{i}.png", image_np)
        cv2.imwrite(f"/home/ccy/cellcycle/test/label{i}.png", label_np)