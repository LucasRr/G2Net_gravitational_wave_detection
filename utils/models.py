import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import timm


class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # input: 2x360x4272
        self.moving_average = nn.Conv2d(2, 2, (1, 48), stride=(1, 48), groups=2, padding=0, bias=None) # 2x360x89
        self.moving_average.weight.data.fill_(np.sqrt(1/48))
        
        self.conv1 = nn.Conv2d(2, 32, (3, 9), stride=(2,4), padding=(0,1))  # 32x180x22
        self.maxpool1 = nn.MaxPool2d((4, 2), (4, 2))                        # 32x45x11
        
        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=(2, 1), padding=(0,1))   # 64x22x11
        self.maxpool2 = nn.MaxPool2d((2, 2), (2, 2), padding=(0,1))            # 64x11x6
        
        self.conv3 = nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1,1))   # 128x11x6
        self.maxpool3 = nn.MaxPool2d((2, 2), (2, 2), padding=(1,0))             # 128x6x3
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128*6*3, 512)
        self.linear2 = nn.Linear(512, 1)
          
    def forward(self, x):
        
        x = self.moving_average(x)
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x



class CNN_v2(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # input: 2x360x4272
        self.moving_average = nn.Conv2d(2, 2, (1, 24), stride=(1, 24), groups=2, padding=0, bias=None) # 2x360x178
        self.moving_average.weight.data.fill_(np.sqrt(1/24))
        
        self.conv1 = nn.Conv2d(2, 32, (3, 13), stride=(1,4), padding=(1,4))  # 32x360x44
        self.maxpool1 = nn.MaxPool2d((2, 1), (2, 1))                         # 32x180x44
        
        self.conv2 = nn.Conv2d(32, 64, (3, 9), stride=(1, 2), padding=(1,4))   # 64x180x22
        self.maxpool2 = nn.MaxPool2d((2, 1), (2, 1), padding=(0,0))            # 64x90x22
        
        self.conv3 = nn.Conv2d(64, 128, (3, 5), stride=(2, 1), padding=(1,2))   # 128x45x22
        self.maxpool3 = nn.MaxPool2d((2, 2), (2, 2), padding=(1,0))             # 128x23x11
        
        self.conv4 = nn.Conv2d(128, 256, (3, 3), stride=(2, 1), padding=(1,1))   # 256x12x11
        self.maxpool4 = nn.MaxPool2d((3, 3), (3, 3), padding=(0,1))             # 256x4x4
        
        self.flatten = nn.Flatten()  # 4096

        self.classifier = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
            )

          
    def forward(self, x):
        
        x = self.moving_average(x)
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)

        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)

        x = F.relu(self.conv4(x))
        x = self.maxpool4(x)
        
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class EfficientNet(nn.Module):
    
    def __init__(self, freeze_blocks=True):
        super().__init__()
        
        # input: 2x360x4279
        self.moving_average = nn.Conv2d(2, 2, (1, 48), stride=(1, 24), groups=2, padding=0, bias=None) # 2x360x89
        self.moving_average.weight.data.fill_(np.sqrt(1/48))
        
        self.conv_stem = nn.Conv2d(2, 32, (3, 11), stride=(2,1), padding=(0,1))  # 32x180x80

        efficientnet_b0 = timm.create_model('efficientnet_b0', checkpoint_path='/Users/lucas/Documents/Code/efficientnet_b0_ra-3dd342df.pth')
        self.blocks = efficientnet_b0.blocks

        self.freeze_blocks = freeze_blocks
        if freeze_blocks:
            for param in self.blocks.parameters():
                param.requires_grad = False
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(nn.Linear(320, 10), nn.Linear(10, 1))
        
    def forward(self, x):
        
        x = self.moving_average(x)
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.adaptive_avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x