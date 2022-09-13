import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self, voltage_num, image_size):
        super(Net, self).__init__()
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
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1, bias=False)

        )


    def forward(self, X):
        Y = self.fc_block(X)
        print(Y)
        Y_reshaped = torch.reshape(Y, (-1, 1, image_size, image_size))
        print(Y_reshaped)
        return Y_reshaped


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

voltage_num = 208
image_size = 4
N = Net(208, 4)
td_volt = torch.randn(2, 208)
N(td_volt)
