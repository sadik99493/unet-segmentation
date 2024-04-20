from torch.utils.data import DataLoader
from data import myDataset
from helper import loadModel,saveModel,get_loaders
import torch
import torch.nn as nn
from model import model
import torch.optim as optim
import albumentations as alb #-->for data augmentation
from albumentations.pytorch import ToTensorV2 #-->for converting images to tensors
from tqdm import tqdm #-->for progress bar during training

#---data directories---
val_imgs = "val_imgs"
val_masks = "val_masks"
training_imgs = "dataset_dir/images"
training_masks = "dataset_dir/masks"

#---hyper-parameters---
lr = 1e-4
batch_size = 20
epochs = 5
img_height = 160
img_width = 240

#---Training function---
def train(dloader , model , criterion , optimizer ):
    model.train()
    loop = tqdm(dloader) #-->to get our data
    for batch_num  , (imgs,masks) in enumerate(loop):
        masks = masks.float().unsqueeze(1) #-->unsqueeze to add channel dimension
        
        #forward prop
        preds = model(imgs)
        loss = criterion(preds,masks/255.0)

        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #update the loss in tqdm bar
        loop.set_postfix(loss = loss.item())

#---Data augmentation---
def augment_and_train():
    training_transforms = alb.Compose( [ alb.Resize(img_height , img_width),
                                        alb.Rotate(limit=35 , p=1.0) , 
                                        alb.HorizontalFlip( p = 0.4 ) ,
                                        alb.VerticalFlip( p = 0.2 ) ,
                                        alb.Normalize( mean = [0.0 , 0.0 , 0.0] ,
                                                      std = [1.0 , 1.0 , 1.0] ,
                                                      max_pixel_value=255.0 ) ,
                                        ToTensorV2() ] )
    
    validation_transforms = alb.Compose( [ alb.Resize(img_height , img_width),
                                            alb.Normalize( mean = [0.0 , 0.0 , 0.0] ,
                                                      std = [1.0 , 1.0 , 1.0] ,
                                                      max_pixel_value=255.0 ) ,
                                            ToTensorV2() ] ) 
    
    #intializing the model
    unet = model()
    #creating loss function
    criterion = nn.BCELoss()
    #creating optimizer
    optimizer = optim.Adam(unet.parameters() , lr = lr )
    #getting the dataloaders
    trainLoader , valLoader = get_loaders( training_imgs, training_masks , val_imgs , val_masks , batch_size , training_transforms , validation_transforms )
    #start training loop
    for epoch in range(epochs):
        train(trainLoader , unet , criterion , optimizer)

        #---save the model after each epoch---
        checkpoint = { "state_dict" : model.state_dict(),
                       "optimizer" : optimizer.state_dict() }
        saveModel(checkpoint)

if __name__ == "__main__":
    augment_and_train()

"""
data = myDataset(img_dir,masks_dir)
unet = model()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(unet.parameters(),lr = 0.001)
epochs = 5
batch_size = 20
for params in unet.parameters():
    params.requires_grad = True

dataLoader = DataLoader(data,batch_size=batch_size,shuffle=True)

unet.train()
for epoch in range(epochs):
    for batch_idx , [img,mask] in enumerate(dataLoader):
        predMask = unet(img)
        loss = criterion(predMask,mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx==20:
            print("loss after 20 batches : ",loss)
    print(f"loss for {epoch}th epoch is :",loss)
"""


