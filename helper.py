import torch
import torchvision
from data import myDataset
from torch.utils.data import DataLoader

def saveModel(model_state,filename = "checkpoint.pth.tar"):
    print("---saving the model---")
    torch.save(model_state , filename)

def loadModel(checkpoint , model):
    print("---loading trained model---")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders( train_imgs , train_masks , val_imgs , val_masks , batch_size , training_transform , validation_transform ):
    
    #---create training loader---
    train_data = myDataset(train_imgs , train_masks , training_transform)
    train_loader = DataLoader(train_data , batch_size , shuffle=True)

    #---create validation loader---
    val_data = myDataset(val_imgs , val_masks , validation_transform)
    val_loader = DataLoader( val_data , batch_size=2 , shuffle=True)

    return train_loader , val_loader

def test_accuracy(loader , model ):
    model.eval()
    dice_score = 0
    corrects = 0
    pixels = 0

    for data,tar in loader:
        preds = model(data)
        preds = torch.sigmoid(preds)
        preds = (preds>0.5).float() #-->make all preds above 0.5 as 1 and rest as 0
        corrects += (preds==tar).sum() #-->find no of pixels where preds and tar match 
        pixels += torch.numel(preds) #-->calculates total number of pixels in preds
        dice_score += 2 * (preds*tar).sum() / ( (preds+tar).sum() + 1e-8)

    print("accuracy ---> " , (corrects/pixels)*100)
    print("dice score ---> " , dice_score/len(loader))

    model.train()
