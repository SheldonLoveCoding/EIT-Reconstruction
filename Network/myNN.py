import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class EITNet(nn.Module):
    def __init__(self, voltage_num, image_size):
        super(EITNet, self).__init__()
        fc1 = nn.Sequential(
            nn.Linear(voltage_num, 4 * image_size * image_size),
            nn.BatchNorm1d(4 * image_size * image_size),
            nn.ReLU(inplace = True) #不用重复申请/释放内存
        )
        fc2 = nn.Sequential(
            nn.Linear(4 * image_size * image_size, image_size * image_size),
            nn.BatchNorm1d(image_size * image_size),
            nn.ReLU(inplace = True)
        )
        fc3 = nn.Sequential(
            nn.Linear(image_size * image_size, image_size * image_size),
            nn.BatchNorm1d(image_size * image_size),
            nn.ReLU(inplace = True)
        )
        self.fc_block = nn.Sequential()
        self.fc_block.add_module('fc1', fc1)
        self.fc_block.add_module('fc2', fc2)
        self.fc_block.add_module('fc3', fc3)

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=(3, 3), padding=(1, 1))
        )

    def forward(self, X, image_size):
        Y = self.fc_block(X)
        # print(Y.shape)
        Y_reshaped = torch.reshape(Y, (-1, 1, image_size, image_size))
        # print(Y_reshaped.shape)
        Y_output = self.conv_block(Y_reshaped)
        # print(Y_output.shape)
        Y_output_reshaped = torch.reshape(Y_output, (-1, 1, image_size, image_size))
        # print(Y_output_reshaped.shape)
        return Y_output_reshaped

