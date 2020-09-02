"""
 Unstructured Mesh adaption of the 
 Implementation (in 2D) of network in:
 http://openaccess.thecvf.com/content_CVPRW_2019/papers/CLIC 2019/Zhou_End-to-end_Optimized_Image_Compression_with_Attention_Mechanism_CVPRW_2019_paper.pdf
"""

import torch
from torch import nn
from UnstructuredCAEDA.nn.pytorch_gdn import GDN
from UnstructuredCAEDA.UnstructuredMesh.RAB1D import RAB1D
from UnstructuredCAEDA.ML_utils import get_device
from UnstructuredCAEDA.AEs.AE_Base import BaseAE
from UnstructuredCAEDA.nn.explore.empty import Empty
import numpy as np

""" 1D Tucodec 2 Linear Layers """ 
class TucodecEncode1D2L(nn.Module):
    def __init__(self, activation_constructor, clusterInputSize, Block, Cstd, DIM, sigmoid=False):
        super(TucodecEncode1D2L, self).__init__()

        device = get_device()
        encode = True

        #downsamples and upsamples
        downsample1 = DownUp1D.downsample1(activation_constructor, Cstd, Cstd)
        upsample1 = DownUp1D.upsample1(activation_constructor, Cstd, Cstd)
        downsample2 = DownUp1D.downsample2(activation_constructor, Cstd, Cstd)
        upsample2 = DownUp1D.upsample2(activation_constructor, Cstd, Cstd)

        # 2 Linear layers
        self.fc1 = nn.Linear(in_features=clusterInputSize, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)

        #main trunk first
        self.conv1 = nn.Conv1d(1, Cstd, kernel_size=3, stride=2, padding=2)
        self.gdn2 = GDN(Cstd, device, not encode)
        self.conv3 = nn.Conv1d(Cstd, Cstd, kernel_size=2, stride=2, padding=0)
        self.gdn4 = GDN(Cstd, device, not encode)
        self.rnab5 = RAB1D(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample1, upsample=upsample1)
        self.conv6 = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=2)
        self.gdn7 = GDN(Cstd, device, not encode)
        self.conv8 = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=2, padding=1)
        self.rnab9 = RAB1D(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample2, upsample=upsample2)

        #multi-res path
        self.convA = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=8)
        self.convB = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=4, padding=0)
        self.convC = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=2, padding=1)

        #final conv
        self.conv10 = nn.Conv1d(4 * Cstd, Cstd, kernel_size=2, stride=2)

    def forward(self, x):
        h, xa, xb, xc, x1, x3, x6, x8 = self.trunk(x)

        ha = self.convA(xa)
        hb = self.convB(xb)
        hc = self.convC(xc)

        inp = torch.cat([h, ha, hb, hc], dim=1) #concat on channel
        x10 = np.shape(inp)
        h = self.conv10(inp)
  
        return h, x1, x3, x6, x8, x10

    def trunk(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        x1 = np.shape(x)
        x = self.conv1(x)
        x = self.gdn2(x)
        xa = x
        x3 = np.shape(xa)
        x = self.conv3(x)
        x = self.gdn4(x)
        xb = x
        x = self.rnab5(x)
        x6 = np.shape(x)
        x = self.conv6(x)
        x = self.gdn7(x)
        xc = x
        x8 = np.shape(xc)
        x = self.conv8(x)
        x = self.rnab9(x)
        return x, xa, xb, xc, x1, x3, x6, x8

class TucodecDecode1D2L(nn.Module):
    def __init__(self, activation_constructor, clusterInputSize, Block, Cstd, DIM, sigmoid=False):
        super(TucodecDecode1D2L, self).__init__()

        device = get_device()
        encode = False

        #downsamples and upsamples
        downsample2 = DownUp1D.downsample2(activation_constructor, Cstd, Cstd)
        upsample2 = DownUp1D.upsample2(activation_constructor, Cstd, Cstd)
        downsample1 = DownUp1D.downsample1(activation_constructor, Cstd, Cstd)
        upsample1 = DownUp1D.upsample1(activation_constructor, Cstd, Cstd)

        #Keep numbering from Encoder

        self.conv10 = nn.ConvTranspose1d( Cstd, Cstd, kernel_size=2, stride=2)

        self.rb10a = Block(encode, activation_constructor, Cstd,)
        self.rb10b = Block(encode, activation_constructor, Cstd,)

        self.rnab9 = RAB1D(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample2, upsample=upsample2)
        self.conv8 = nn.ConvTranspose1d(Cstd, Cstd, kernel_size=3, stride=2, padding=1, output_padding=0)



        self.gdn7 = GDN(Cstd, device, encode)
        self.conv6 = nn.ConvTranspose1d(Cstd, Cstd, kernel_size=3, stride=2, output_padding=1)
        self.rnab5 = RAB1D(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample1, upsample=upsample1)
        self.gdn4 = GDN(Cstd, device, encode)

        self.conv3 = nn.ConvTranspose1d(Cstd, Cstd, kernel_size=2, stride=2, padding=0, output_padding=1)
        self.gdn2 = GDN(Cstd, device, encode)
        self.conv1 = nn.ConvTranspose1d(Cstd, 1, kernel_size=3, stride=2, padding=2, output_padding=0)
        
        self.fc2 = nn.Linear(in_features=256, out_features=512)
        self.fc1 = nn.Linear(in_features=512, out_features=clusterInputSize)


    def forward(self, x, x1, x3, x6, x8, x10):
        x = self.conv10(x, x10)
        x = self.rb10a(x)
        x = self.rb10b(x)

        x = self.rnab9 (x)
        x = self.conv8 (x, x8)

        x = self.gdn7(x)
        x = self.conv6(x, x6)
        x = self.rnab5(x)
        x = self.gdn4 (x)

        x = self.conv3(x, x3)
        x = self.gdn2(x)
        x = self.conv1(x, x1)

        x = self.fc2(x)
        x = self.fc1(x)
        
        return x

""" 1D Tucodec 4 Linear Layers """
 
class TucodecEncode1D4L(nn.Module):
    def __init__(self, activation_constructor, clusterInputSize, Block, Cstd, DIM, sigmoid=False):
        super(TucodecEncode1D4L, self).__init__()

        device = get_device()
        encode = True

        #downsamples and upsamples
        downsample1 = DownUp1D.downsample1(activation_constructor, Cstd, Cstd)
        upsample1 = DownUp1D.upsample1(activation_constructor, Cstd, Cstd)
        downsample2 = DownUp1D.downsample2(activation_constructor, Cstd, Cstd)
        upsample2 = DownUp1D.upsample2(activation_constructor, Cstd, Cstd)

        # 4 Linear Layers
        self.fc1 = nn.Linear(in_features=clusterInputSize, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=128)

        #main trunk first
        self.conv1 = nn.Conv1d(1, Cstd, kernel_size=3, stride=2, padding=1)
        self.gdn2 = GDN(Cstd, device, not encode)
        self.conv3 = nn.Conv1d(Cstd, Cstd, kernel_size=2, stride=2, padding=0)
        self.gdn4 = GDN(Cstd, device, not encode)
        self.rnab5 = RAB1D(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample1, upsample=upsample1)
        self.conv6 = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=2)
        self.gdn7 = GDN(Cstd, device, not encode)
        self.conv8 = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=2, padding=1)
        self.rnab9 = RAB1D(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample2, upsample=upsample2)

        #multi-res path
        self.convA = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=8)
        self.convB = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=4, padding=0)
        self.convC = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=2, padding=1)

        #final conv
        self.conv10 = nn.Conv1d(4 * Cstd, Cstd, kernel_size=2, stride=2)

    def forward(self, x):
        h, xa, xb, xc, x1, x3, x6, x8 = self.trunk(x)

        ha = self.convA(xa)
        hb = self.convB(xb)
        hc = self.convC(xc)

        inp = torch.cat([h, ha, hb, hc], dim=1) #concat on channel
        x10 = np.shape(inp)
        h = self.conv10(inp)

        return h, x1, x3, x6, x8, x10

    def trunk(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        x1 = np.shape(x)
        x = self.conv1(x)
        x = self.gdn2(x)
        xa = x
        x3 = np.shape(xa)
        x = self.conv3(x)
        x = self.gdn4(x)
        xb = x
        x = self.rnab5(x)
        x6 = np.shape(x)
        x = self.conv6(x)
        x = self.gdn7(x)
        xc = x
        x8 = np.shape(xc)
        x = self.conv8(x)
        x = self.rnab9(x)
        return x, xa, xb, xc, x1, x3, x6, x8

class TucodecDecode1D4L(nn.Module):
    def __init__(self, activation_constructor, clusteringInputSize, Block, Cstd, DIM, sigmoid=False):
        super(TucodecDecode1D4L, self).__init__()

        device = get_device()
        encode = False

        #downsamples and upsamples
        downsample2 = DownUp1D.downsample2(activation_constructor, Cstd, Cstd)
        upsample2 = DownUp1D.upsample2(activation_constructor, Cstd, Cstd)
        downsample1 = DownUp1D.downsample1(activation_constructor, Cstd, Cstd)
        upsample1 = DownUp1D.upsample1(activation_constructor, Cstd, Cstd)

        #Keep numbering from Encoder

        self.conv10 = nn.ConvTranspose1d( Cstd, Cstd, kernel_size=2, stride=2)

        self.rb10a = Block(encode, activation_constructor, Cstd,)
        self.rb10b = Block(encode, activation_constructor, Cstd,)

        self.rnab9 = RAB1D(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample2, upsample=upsample2)
        self.conv8 = nn.ConvTranspose1d(Cstd, Cstd, kernel_size=3, stride=2, padding=1, output_padding=0)



        self.gdn7 = GDN(Cstd, device, encode)
        self.conv6 = nn.ConvTranspose1d(Cstd, Cstd, kernel_size=3, stride=2, output_padding=1)
        self.rnab5 = RAB1D(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample1, upsample=upsample1)
        self.gdn4 = GDN(Cstd, device, encode)

        self.conv3 = nn.ConvTranspose1d(Cstd, Cstd, kernel_size=2, stride=2, padding=0, output_padding=1)
        self.gdn2 = GDN(Cstd, device, encode)
        self.conv1 = nn.ConvTranspose1d(Cstd, 1, kernel_size=3, stride=2, padding=1, output_padding=0)

        self.fc4 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1024)
        self.fc1 = nn.Linear(in_features=1024, out_features=clusterInputSize)

    def forward(self, x, x1, x3, x6, x8, x10):
        x = self.conv10(x, x10)
        x = self.rb10a(x)
        x = self.rb10b(x)

        x = self.rnab9 (x)
        x = self.conv8 (x, x8)

        x = self.gdn7(x)
        x = self.conv6(x, x6)
        x = self.rnab5(x)
        x = self.gdn4 (x)

        x = self.conv3(x, x3)
        x = self.gdn2(x)
        x = self.conv1(x, x1)
       
        x = self.fc4(x)
        x = self.fc3(x)
        x = self.fc2(x)
        x = self.fc1(x)

        return x

""" 1D Tucodec 0 Linear Layers """
 
class TucodecEncode1D0L(nn.Module):
    def __init__(self, activation_constructor, Block, Cstd, DIM, sigmoid=False):
        super(TucodecEncode1D0L, self).__init__()

        device = get_device()
        encode = True

        #downsamples and upsamples
        downsample1 = DownUp1D.downsample1(activation_constructor, Cstd, Cstd)
        upsample1 = DownUp1D.upsample1(activation_constructor, Cstd, Cstd)
        downsample2 = DownUp1D.downsample2(activation_constructor, Cstd, Cstd)
        upsample2 = DownUp1D.upsample2(activation_constructor, Cstd, Cstd)

        #main trunk first
        self.conv1 = nn.Conv1d(1, Cstd, kernel_size=3, stride=2, padding=1)
        self.gdn2 = GDN(Cstd, device, not encode)
        self.conv3 = nn.Conv1d(Cstd, Cstd, kernel_size=2, stride=2, padding=0)
        self.gdn4 = GDN(Cstd, device, not encode)
        self.rnab5 = RAB1D(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample1, upsample=upsample1)
        self.conv6 = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=2)
        self.gdn7 = GDN(Cstd, device, not encode)
        self.conv8 = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=2, padding=1)
        self.rnab9 = RAB1D(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample2, upsample=upsample2)

        #multi-res path
        self.convA = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=8)
        self.convB = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=4, padding=0)
        self.convC = nn.Conv1d(Cstd, Cstd, kernel_size=3, stride=2, padding=1)

        #final conv
        self.conv10 = nn.Conv1d(4 * Cstd, Cstd, kernel_size=2, stride=2)

    def forward(self, x):
        h, xa, xb, xc, x1, x3, x6, x8 = self.trunk(x)

        ha = self.convA(xa)
        hb = self.convB(xb)
        hc = self.convC(xc)

        inp = torch.cat([h, ha, hb, hc], dim=1) #concat on channel
        x10 = np.shape(inp)
        h = self.conv10(inp)

        return h, x1, x3, x6, x8, x10

    def trunk(self, x):

        x1 = np.shape(x)
        x = self.conv1(x)
        x = self.gdn2(x)
        xa = x
        x3 = np.shape(xa)
        x = self.conv3(x)
        x = self.gdn4(x)
        xb = x
        x = self.rnab5(x)
        x6 = np.shape(x)
        x = self.conv6(x)
        x = self.gdn7(x)
        xc = x
        x8 = np.shape(xc)
        x = self.conv8(x)
        x = self.rnab9(x)
        return x, xa, xb, xc, x1, x3, x6, x8

class TucodecDecode1D0L(nn.Module):
    def __init__(self, activation_constructor, Block, Cstd, DIM, sigmoid=False):
        super(TucodecDecode1D0L, self).__init__()

        device = get_device()
        encode = False

        #downsamples and upsamples
        downsample2 = DownUp1D.downsample2(activation_constructor, Cstd, Cstd)
        upsample2 = DownUp1D.upsample2(activation_constructor, Cstd, Cstd)
        downsample1 = DownUp1D.downsample1(activation_constructor, Cstd, Cstd)
        upsample1 = DownUp1D.upsample1(activation_constructor, Cstd, Cstd)

        #Keep numbering from Encoder

        self.conv10 = nn.ConvTranspose1d( Cstd, Cstd, kernel_size=2, stride=2)

        self.rb10a = Block(encode, activation_constructor, Cstd,)
        self.rb10b = Block(encode, activation_constructor, Cstd,)

        self.rnab9 = RAB1D(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample2, upsample=upsample2)
        self.conv8 = nn.ConvTranspose1d(Cstd, Cstd, kernel_size=3, stride=2, padding=1, output_padding=0)



        self.gdn7 = GDN(Cstd, device, encode)
        self.conv6 = nn.ConvTranspose1d(Cstd, Cstd, kernel_size=3, stride=2, output_padding=1)
        self.rnab5 = RAB1D(encode, activation_constructor, Cstd, sigmoid, Block,
                        downsample=downsample1, upsample=upsample1)
        self.gdn4 = GDN(Cstd, device, encode)

        self.conv3 = nn.ConvTranspose1d(Cstd, Cstd, kernel_size=2, stride=2, padding=0, output_padding=1)
        self.gdn2 = GDN(Cstd, device, encode)
        self.conv1 = nn.ConvTranspose1d(Cstd, 1, kernel_size=3, stride=2, padding=1, output_padding=0)

    def forward(self, x, x1, x3, x6, x8, x10):
        x = self.conv10(x, x10)
        x = self.rb10a(x)
        x = self.rb10b(x)

        x = self.rnab9 (x)
        x = self.conv8 (x, x8)

        x = self.gdn7(x)
        x = self.conv6(x, x6)
        x = self.rnab5(x)
        x = self.gdn4 (x)

        x = self.conv3(x, x3)
        x = self.gdn2(x)
        x = self.conv1(x, x1)
       
        return x

class DownUp1D:
    @staticmethod
    def downsample1(activation_constructor, Cin, channel_small):
        """First RAB downsample"""
        conv1 = nn.Conv1d(Cin, Cin, kernel_size=3, stride=2)
        conv2 = nn.Conv1d(Cin, Cin, kernel_size=3, stride=2, padding=0)
        conv3 = nn.Conv1d(Cin, Cin, kernel_size=3, stride=2, padding=0)
        return nn.Sequential(conv1, activation_constructor(Cin, False), #Empty("d", 1),
                            conv2, activation_constructor(Cin, False), #Empty("d", 2),
                            conv3, )#Empty("d", 3),)
    @staticmethod
    def upsample1(activation_constructor, Cin, channel_small):
        "First RAB upsample"
        conv1 = nn.ConvTranspose1d(Cin, Cin, kernel_size=3, stride=2, padding=0)
        conv2 = nn.ConvTranspose1d(Cin, Cin, kernel_size=3, stride=2, padding=0)
        conv3 = nn.ConvTranspose1d(Cin, Cin, kernel_size=3, stride=2)
        return nn.Sequential(conv1, activation_constructor(Cin, False), #Empty("u", 1),
                            conv2, activation_constructor(Cin, False), #Empty("u", 2),
                            conv3, ) #Empty("u", 3))

    @staticmethod
    def downsample2(activation_constructor, Cin, channel_small):
        """Second RAB downsample"""
        # print("Gets downsampled2 called")
        conv1 = nn.Conv1d(Cin, Cin, kernel_size=2, stride=1)
        conv2 = nn.Conv1d(Cin, Cin, kernel_size=3, stride=2, padding=0)
        conv3 = nn.Conv1d(Cin, Cin, kernel_size=2, stride=1, padding=0)
        return nn.Sequential(conv1, activation_constructor(Cin, False),
                            conv2, activation_constructor(Cin, False),
                            conv3, )
    @staticmethod
    def upsample2(activation_constructor, Cin, channel_small):
        """Second RAB upsample"""
        conv1 = nn.ConvTranspose1d(Cin, Cin, kernel_size=2, stride=1, padding=0)
        conv2 = nn.ConvTranspose1d(Cin, Cin, kernel_size=3, stride=2, padding=0)
        conv3 = nn.ConvTranspose1d(Cin, Cin, kernel_size=2, stride=1,)
        return nn.Sequential(conv1, activation_constructor(Cin, False),
                            conv2, activation_constructor(Cin, False),
                            conv3, )
