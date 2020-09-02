""" Code from https://github.com/julianmack/Data_Assimilation with only minor modifications """

"""
Implementation of  <https://arxiv.org/pdf/1608.06993.pdf> for 3D case.

The following code is heavily based on the DenseNet implementation
in torchvision.models.DenseNet - it was not possible to use the original
implementation directly because:
    1) It is for 2d input rather than 3D
    2) The order of convolution, BN, Activation of (BN -> ReLU -> conv -> BN -> ...) is not
    typically used anymore (Now BN -> conv -> ReLU -> BN) is more common


LICENCE: this is licenced under the BSD 3-Clause License
see: https://github.com/pytorch/vision/blob/master/LICENSE
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from UnstructuredCAEDA.nn.res import ResNextBlock


class _DenseBlock(nn.Module):
    def __init__(self, encode, activation_constructor, Cin, growth_rate, Csmall,
                    dense_layers, Block=ResNextBlock, residual=False):
        super(_DenseBlock, self).__init__()
        self.residual = residual
        for i in range(dense_layers):

            layer = Block(encode, activation_constructor,
                Cin = Cin + i * growth_rate,
                channel_small = Csmall,
                Cout = growth_rate,
                residual=False,
            )

            self.add_module('denselayer%d' % (i + 1), layer)

        squeeze =  nn.Conv3d(Cin + (i + 1) * growth_rate, Cin, kernel_size=(1, 1, 1), stride=(1,1,1))

        self.add_module('denseSqueeze', squeeze)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            if name == "denseSqueeze":
                continue
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        h = torch.cat(features, 1)
        h = self.denseSqueeze(h)
        if self.residual:
            h = h + init_features
        return h


