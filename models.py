## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # Input shape: (224, 224)
        # Output shape: (136, 1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3)  # (112, 112)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (28, 28)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # (28, 28)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (7, 7)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # (28, 28)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (14, 14)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # (14, 14)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (7, 7)
        
#         self.avgpool = nn.AvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(256 * 7 * 7, 4096)
        self.dropout = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(4096, 136)

        
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.maxpool3(F.relu(self.conv3(x)))
        x = self.maxpool4(F.relu(self.conv4(x)))
#         x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
