from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np

class myDataset(Dataset):
    def __init__(self , img_dir , mask_dir , transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)
        self.masks = os.listdir(mask_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) :
        img_path = os.path.join(self.img_dir,self.images[index])
        mask_path = os.path.join(self.mask_dir,self.masks[index])
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"),dtype=np.float32) #-->convert to float
        mask[mask==255.0] = 1.0 #-->convert white pixel values from 255 to 1 | black pixel values will be 0 
        augments = self.transform(image = img , mask = mask)
        final_img = augments['image']
        final_mask = augments['mask']
        return final_img , final_mask