
"""
1D Implementation of Residual Blocks originally in 3D from:
https://github.com/julianmack/Data_Assimilation/blob/master/src/VarDACAE/nn/res.py
"""

import torch
from torch import nn
from UnstructuredCAEDA.nn import init
from torch.nn.parameter import Parameter
from UnstructuredCAEDA.nn.CBAM import CBAM

class ResVanilla1D(nn.Module):
    """Standard residual block (slightly adapted to our use case)
    """
    def __init__(self, encode, activation_constructor, Cin, channel_small=None,
                    down_sf=4, Cout=None, residual=True):
        super(ResVanilla1D, self).__init__()
        self.residual = residual
        if Cout is None:
            Cout = Cin
        if channel_small is None:
            channel_small = Cin
        conv1 = nn.Conv1d(Cin, channel_small, kernel_size=3, stride=1, padding=1)
        conv2 = nn.Conv1d(channel_small, Cout, kernel_size=3, stride=1, padding=1)

        #Initializations
        init.conv(conv1.weight, activation_constructor)
        init.conv(conv2.weight, activation_constructor)

        #ADD batch norms automatically
        self.ResLayers = nn.Sequential(conv1,
            activation_constructor(channel_small, not encode), nn.BatchNorm2d(channel_small), conv2,
            activation_constructor(Cout, not encode))

    def forward(self, x):
        h = self.ResLayers(x)
        if self.residual:
            h = h + x
        return h

class ResNextBlock1D(nn.Module):
    """Single res-block from arXiv:1611.05431v2 

    It is really just a standard res_block with sqeezed 1x1 convs
    at input and output
    """
    def __init__(self, encode, activation_constructor, Cin, channel_small=None,
                    down_sf=4, Cout=None, residual=True):
        super(ResNextBlock1D, self).__init__()
        self.residual = residual
        if Cout is None:
            Cout = Cin

        if not channel_small:
            #Minimum of 4 channels
            channel_small = (Cin // down_sf) if (Cin // down_sf > 0) else 4

        conv1x1_1 = nn.Conv1d(Cin, channel_small, kernel_size=1, stride=1)
        conv3x3 = nn.Conv1d(channel_small, channel_small, kernel_size=3, stride=1, padding=1)
        conv1x1_2 = nn.Conv1d(channel_small, Cout, kernel_size=1, stride=1)

        #Initializations
        init.conv(conv1x1_1.weight, activation_constructor)
        init.conv(conv3x3.weight, activation_constructor)
        init.conv(conv1x1_2.weight, activation_constructor)

        #ADD batch norms automatically
        self.ResLayers = nn.Sequential(conv1x1_1,
            activation_constructor(channel_small, not encode), nn.BatchNorm1d(channel_small), conv3x3,
            activation_constructor(channel_small, not encode), nn.BatchNorm1d(channel_small),
            conv1x1_2, activation_constructor(Cout, not encode))

    def forward(self, x):
        h = self.ResLayers(x)
        if self.residual:
            h = h + x
        return h

class ResVanilla2D(nn.Module):
    """Standard residual block (slightly adapted to our use case)
    """
    def __init__(self, encode, activation_constructor, Cin, channel_small=None,
                    down_sf=4, Cout=None, residual=True):
        super(ResVanilla2D, self).__init__()
        self.residual = residual
        if Cout is None:
            Cout = Cin
        if channel_small is None:
            channel_small = Cin
        conv1 = nn.Conv2d(Cin, channel_small, kernel_size=(3, 3), stride=(1,1), padding=(1,1))
        conv2 = nn.Conv2d(channel_small, Cout, kernel_size=(3, 3), stride=(1,1), padding=(1,1))

        #Initializations
        init.conv(conv1.weight, activation_constructor)
        init.conv(conv2.weight, activation_constructor)

        #ADD batch norms automatically
        self.ResLayers = nn.Sequential(conv1,
            activation_constructor(channel_small, not encode), nn.BatchNorm2d(channel_small), conv2,
            activation_constructor(Cout, not encode))

    def forward(self, x):
        h = self.ResLayers(x)
        if self.residual:
            h = h + x
        return h

class ResNextBlock2D(nn.Module):
    """Single res-block from arXiv:1611.05431v2

    It is really just a standard res_block with sqeezed 1x1 convs
    at input and output
    """
    def __init__(self, encode, activation_constructor, Cin, channel_small=None,
                    down_sf=4, Cout=None, residual=True):
        super(ResNextBlock2D, self).__init__()
        self.residual = residual
        if Cout is None:
            Cout = Cin

        if not channel_small:
            #Minimum of 4 channels
            channel_small = (Cin // down_sf) if (Cin // down_sf > 0) else 4

        conv1x1_1 = nn.Conv2d(Cin, channel_small, kernel_size=(1, 1), stride=(1,1))
        conv3x3 = nn.Conv2d(channel_small, channel_small, kernel_size=(3, 3), stride=(1,1), padding=(1,1))
        conv1x1_2 = nn.Conv2d(channel_small, Cout, kernel_size=(1, 1), stride=(1,1))

        #Initializations
        init.conv(conv1x1_1.weight, activation_constructor)
        init.conv(conv3x3.weight, activation_constructor)
        init.conv(conv1x1_2.weight, activation_constructor)

        #ADD batch norms automatically
        self.ResLayers = nn.Sequential(conv1x1_1,
            activation_constructor(channel_small, not encode), nn.BatchNorm2d(channel_small), conv3x3,
            activation_constructor(channel_small, not encode), nn.BatchNorm2d(channel_small),
            conv1x1_2, activation_constructor(Cout, not encode))
            #nn.BatchNorm3d(Cin))

    def forward(self, x):
        h = self.ResLayers(x)
        if self.residual:
            h = h + x
        return h