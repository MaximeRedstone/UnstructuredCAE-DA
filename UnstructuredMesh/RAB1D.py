
"""
1D Implementation of Residual Attention blocks originally in 3D from:
https://github.com/julianmack/Data_Assimilation/blob/master/src/VarDACAE/nn/RAB.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from UnstructuredCAEDA.nn.res import ResNextBlock, ResVanilla
from UnstructuredCAEDA.nn.explore.empty import Empty
from UnstructuredCAEDA.nn.helpers import get_activation
import numpy as np

from UnstructuredCAEDA.UnstructuredMesh.HelpersUnstructuredMesh import *

class RAB1D(nn.Module):

    def __init__(self, encode, activation_constructor, Cin, sigmoid=True,
                    Block = ResVanilla, channel_small=None,
                    down_sf=4, residual=True, downsample=None,
                    upsample=None, ):
        super(RAB1D, self).__init__()
        if downsample is not None:
            assert upsample is not None
        self.sigmoid = sigmoid
        self.residual = residual
    
        #init trunk: 3 res blocks
        self.trunk = nn.Sequential()
        for i in range(3):
            res = Block(encode, activation_constructor, Cin, channel_small)
            self.trunk.add_module('res%d' % (i), res)

        #init mask
        self.mask = nn.Sequential()
        for i in range(2):
            res = Block(encode, activation_constructor, Cin, channel_small)
            self.mask.add_module('res%d' % (i), res)
        if not downsample:
            downsample = self.__build_downsample(encode, activation_constructor, Cin, channel_small)
        self.mask.add_module('downsample', downsample)
        
        for i in range(2, 4):
            res = Block(encode, activation_constructor, Cin, channel_small)
            self.mask.add_module('res%d' % (i), res)
        if not upsample:
            upsample = self.__build_upsample(encode, activation_constructor, Cin, channel_small)
        self.mask.add_module('upsample', upsample)

        for i in range(4, 6):
            res = Block(encode, activation_constructor, Cin, channel_small)
            self.mask.add_module('res%d' % (i), res)

        self.mask.add_module('conv1x1', nn.Conv1d(Cin, Cin, kernel_size=1, padding=0))

    def __build_downsample(self, encode, activation_constructor, Cin, channel_small):
        """This downsample is specific to out input size in this case of
        C, x, y, z = 32, 11, 11, 3"""
        conv1 = nn.Conv1d(Cin, Cin, kernel_size=3, stride=2)
        conv2 = nn.Conv1d(Cin, Cin, kernel_size=3, stride=2) #, padding=(1, 1))
        conv3 = nn.Conv1d(Cin, Cin, kernel_size=3, stride=1)
        return nn.Sequential(conv1, activation_constructor(Cin, False),
                            conv2, activation_constructor(Cin, False),
                            conv3, )

    def __build_upsample(self, encode, activation_constructor, Cin, channel_small):
        """This downsample is specific to out input size in this case of
        C, x, y, z = 32, 11, 11, 3"""
        conv1 = nn.ConvTranspose1d(Cin, Cin, kernel_size=3, stride=1)
        conv2 = nn.ConvTranspose1d(Cin, Cin, kernel_size=3, stride=2) #, padding=(1, 1))
        conv3 = nn.ConvTranspose1d(Cin, Cin, kernel_size=3, stride=2)

        return nn.Sequential(conv1, activation_constructor(Cin, True),
                            conv2, activation_constructor(Cin, True),
                            conv3)

    def forward(self, x):
        mask = self.mask(x)
        if self.sigmoid:
            mask = torch.sigmoid(mask)
        trunk = self.trunk(x)

        trunk, mask = matchDimensions1D(trunk, mask) # Add padding to ensure overlapping on trunk and mask
        h = trunk * mask

        if self.residual:
            res = h + x
        return res
