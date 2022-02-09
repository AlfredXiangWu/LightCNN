# -*- coding: utf-8 -*-
# @Author: Alfred Xiang Wu
# @Date:   2022-02-09 14:45:31
# @Breif: 
# @Last Modified by:   Alfred Xiang Wu
# @Last Modified time: 2022-02-09 14:48:34

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):    
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class resblock_v1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock_v1, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class network(nn.Module):
    def __init__(self, block, layers):
        super(network, self).__init__()

        self.conv1 = mfm(3, 48, 3, 1, 1)

        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.conv2  = mfm(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.conv3  = mfm(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.conv4  = mfm(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.conv5  = mfm(128, 128, 3, 1, 1)

        self.fc = nn.Linear(8*8*128, 256)
        nn.init.normal_(self.fc.weight, std=0.001)
            
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, label=None):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.conv4(x)
        x = self.block4(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = torch.flatten(x, 1)
        fc = self.fc(x)

        return fc

def LightCNN_V4(cfg):
    model = network(resblock_v1, [1, 2, 3, 4])
    return model