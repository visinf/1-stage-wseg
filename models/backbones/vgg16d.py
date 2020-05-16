import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from models.backbones.base_net import BaseNet

class VGG16(BaseNet):
    def __init__(self, fc6_dilation = 1):
        super(VGG16, self).__init__()

        self.conv1_1 = nn.Conv2d(3,64,3,padding = 1)
        self.conv1_2 = nn.Conv2d(64,64,3,padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.conv2_1 = nn.Conv2d(64,128,3,padding = 1)
        self.conv2_2 = nn.Conv2d(128,128,3,padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.conv3_1 = nn.Conv2d(128,256,3,padding = 1)
        self.conv3_2 = nn.Conv2d(256,256,3,padding = 1)
        self.conv3_3 = nn.Conv2d(256,256,3,padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.conv4_1 = nn.Conv2d(256,512,3,padding = 1)
        self.conv4_2 = nn.Conv2d(512,512,3,padding = 1)
        self.conv4_3 = nn.Conv2d(512,512,3,padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding=1)
        self.conv5_1 = nn.Conv2d(512,512,3,padding = 2, dilation = 2)
        self.conv5_2 = nn.Conv2d(512,512,3,padding = 2, dilation = 2)
        self.conv5_3 = nn.Conv2d(512,512,3,padding = 2, dilation = 2)

        self.fc6 = nn.Conv2d(512,1024, 3, padding = fc6_dilation, dilation = fc6_dilation)

        self.drop6 = nn.Dropout2d(p=0.5)
        self.fc7 = nn.Conv2d(1024, 1024, 1)

        # fixing the parameters
        self._fix_params([self.conv1_1, self.conv1_2])

    def fan_out(self):
        return 1024

    def forward(self, x):
        return self.forward_as_dict(x)['conv6']

    def forward_as_dict(self, x):

        x = F.relu(self.conv1_1(x), inplace=True)
        x = F.relu(self.conv1_2(x), inplace=True)
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x), inplace=True)
        x = F.relu(self.conv2_2(x), inplace=True)
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x), inplace=True)
        x = F.relu(self.conv3_2(x), inplace=True)
        x = F.relu(self.conv3_3(x), inplace=True)
        conv3 = x

        x = self.pool3(x)

        x = F.relu(self.conv4_1(x), inplace=True)
        x = F.relu(self.conv4_2(x), inplace=True)
        x = F.relu(self.conv4_3(x), inplace=True)

        x = self.pool4(x)

        x = F.relu(self.conv5_1(x), inplace=True)
        x = F.relu(self.conv5_2(x), inplace=True)
        x = F.relu(self.conv5_3(x), inplace=True)

        x = F.relu(self.fc6(x), inplace=True)
        x = self.drop6(x)
        x = F.relu(self.fc7(x), inplace=True)

        conv6 = x

        return dict({'conv3': conv3, 'conv6': conv6})
