import torch
import torch.nn as nn

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        # conv layer
        C, H, W = 3, 32, 32
        # my configuration
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # conv layers 1
        self.conv1 = nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # conv layers 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p=0.05)
        
        # conv layers 3
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(p=0.1)
        
        # fully connected layers
        self.fc1 = nn.Linear(4*4*128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(512, 9)


    def forward(self, x):
        outs = None
        #(N, C, H, W) = x.shape
        #print(C, H, W)
        # conv layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # conv layer 2
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        
        # conv layer 3
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = x.view(-1, 4*4*128)
        # fully conected layer 
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout3(x)
        outs = self.fc3(x)

        return outs