import torch
import torch.nn as nn
import torchvision.transforms.functional as f

class DoubleConvolution(nn.Module):
    def __init__(self,inp,out):
        super(DoubleConvolution,self).__init__()
        self.horizontalStep = nn.Sequential( nn.Conv2d(in_channels=inp,out_channels=out,kernel_size= 3, stride=1,padding=1,bias=False),
                                     nn.BatchNorm2d(out),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels=out,out_channels=out,kernel_size=3,stride = 1,padding=1,bias=False),
                                     nn.BatchNorm2d(out),
                                     nn.ReLU(inplace=True))
    def forward(self,x):
        return self.horizontalStep(x)
    
class model(nn.Module):
    def __init__(self, in_channels = 3 , out_channels = 1 , dims = [64,128,256,512]):
        super().__init__()
        self.downConvs = nn.ModuleList()
        self.upConvs = nn.ModuleList()
        self.pooling = nn.MaxPool2d(2 , 2)

        #---DOWN SAMPLING CONVOLUTIONS----
        for out_dim in dims : 
            self.downConvs.append(DoubleConvolution(in_channels, out_dim))
            in_channels = out_dim

        #---UP SAMPLING CONVOLUTIONS------
        for in_channel in dims[::-1]:
            self.upConvs.append( nn.ConvTranspose2d(in_channel*2 , in_channel , 2 , 2))
            self.upConvs.append( DoubleConvolution(in_channel*2 , in_channel ))

        #---BOTTLENECK------
        self.bottleneck = DoubleConvolution(dims[-1] , dims[-1]*2)
        self.final = nn.Conv2d(dims[0] , out_channels , kernel_size=1)

    def forward(self , x):
        skipCons = []
        #----APPLY DOWN CONVOLUITONS------
        for downConv in self.downConvs:
            x = downConv(x)    
            skipCons.append(x)  #-->store it for skip connections
            x = self.pooling(x)

        #-----APPLY BOTTLENECK----
        x = self.bottleneck(x)
        skipCons = skipCons[::-1] #-->reversing skip cons for easy access

        #----APPLY UP CONVOLUTIONS
        for step in range(0 , len(self.upConvs), 2):
            x = self.upConvs[step](x)
            skipCon = skipCons[step//2]

            if x.shape != skipCon.shape:
                x = f.resize(x , skipCon.shape[2:])

            x = torch.cat( (x , skipCon) , dim=1)
            x = self.upConvs[step+1](x)

        return torch.sigmoid(self.final(x))



        
if __name__ == "__main__":
    out = torch.tensor([[[1,2,3],[2,3,4]],[[1,2,3],[2,3,4]],[[4,5,6],[5,6,7]]])
    print(out/7)