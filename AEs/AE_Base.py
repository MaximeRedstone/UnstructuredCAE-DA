""" Code from https://github.com/julianmack/Data_Assimilation with only minor modifications """

import torch.nn as nn
import torch
import numpy as np

class BaseAE(nn.Module):
    """Base AE class which all should inherit from
    The following instance variables must be instantiated in __init__:
        self.layers_encode - an nn.ModuleList of all encoding layers in the network
        self.layers_decode - an nn.ModuleList of all decoding layers in the network
        self.act_fn - the activation function to use in between layers
    If known, the following instance variables *should* be instantiated in __init__:
        self.latent_sz - a tuple containing the latent size of the system
                        (NOT including the batch number).
                        e.g. if latent.shape = (M x Cout x nx x ny x nz) then
                        latent_size = (Cout, nx, ny, nz)
    """
    def forward(self, x):
        self.__check_instance_vars()
        x, x1, x3, x6, x8, x10 = self.encode(x)
        x = self.decode(x, x1, x3, x6, x8, x10)

        return x

    def encode(self, x):
        x = self.__maybe_convert_to_batched(x, CASE_3D=False, CASE_2D=False, CASE_1D=True) #1D Case
        # x = self.__maybe_convert_to_batched(x, CASE_3D=False, CASE_2D=True, CASE_1D=False) #2D Case
        # x = self.__maybe_convert_to_batched(x, CASE_3D=True, CASE_2D=False, CASE_1D=False) #3D Case
        layers = self.layers_encode
        for layer in layers[:-1]:
            x = self.act_fn(layer(x))
        x, x1, x3, x6, x8, x10 = layers[-1](x) #no activation function for latent space
        x = self.__flatten_encode(x)
        x = self.__maybe_convert_to_non_batched(x, CASE_3D=False, CASE_2D=False, CASE_1D=True) #1D Case
        # x = self.__maybe_convert_to_non_batched(x, CASE_3D=False, CASE_2D=True, CASE_1D=False) #2D Case
        # x = self.__maybe_convert_to_non_batched(x, CASE_3D=True, CASE_2D=False, CASE_1D=False) #3D Case
        return x, x1, x3, x6, x8, x10

    def decode(self, x, x1, x3, x6, x8, x10, latent_sz=None):
        x = self.__maybe_convert_to_batched(x, CASE_3D=False, CASE_2D=False, CASE_1D=True) #1D Case
        # x = self.__maybe_convert_to_batched(x, CASE_3D=False, CASE_2D=True, CASE_1D=False) #2D Case
        # x = self.__maybe_convert_to_batched(x, CASE_3D=True, CASE_2D=False, CASE_1D=False) #3D Case
        x = self.__unflatten_decode(x, latent_sz)
        layers = self.layers_decode
        for layer in layers[:-1]:
            x = self.act_fn(layer(x, ))
        x = layers[-1](x, x1, x3, x6, x8, x10) #no activation function for output
        x = self.__maybe_convert_to_non_batched(x, CASE_3D=False, CASE_2D=False, CASE_1D=True) #1D Case
        # x = self.__maybe_convert_to_non_batched(x, CASE_3D=False, CASE_2D=True, CASE_1D=False) #2D Case
        # x = self.__maybe_convert_to_non_batched(x, CASE_3D=True, CASE_2D=False, CASE_1D=False) #3D Case
        return x

    def __check_instance_vars(self):
        try:
            decode = self.layers_decode
            encode = self.layers_encode
        except:
            raise ValueError("Must init model with instance variables layers_decode and layers_encode")
        assert isinstance(decode, (nn.ModuleList, nn.Sequential)), "model.layers_decode must be of type nn.ModuleList"
        assert isinstance(encode, (nn.ModuleList, nn.Sequential)), "model.layers_encode must be of type nn.ModuleList"

    def __flatten_encode(self, x):
        """Flattens input after encoding and saves latent_sz.
        NOTE: all inputs x will be batched"""

        self.latent_sz = x.shape[1:]

        x = torch.flatten(x, start_dim=1) #start at dim = 1 since batched input

        return x

    def __unflatten_decode(self, x, latent_sz=None):
        """Unflattens decoder input before decoding.
        NOTE: If the AE has not been used for an encoding, it is necessary to pass
        the desired latent_sz.
        NOTE: all inputs x will be batched"""
        if latent_sz == None:
            if hasattr(self, "latent_sz"):
                latent_sz = self.latent_sz
            else:
                latent_sz = None
        if latent_sz == None:
            raise ValueError("No latent_sz provided to decoder and encoder not run")

        self.latent_sz = latent_sz
        size = (self.batch_sz,) + tuple(self.latent_sz)

        x = x.view(size)

        return x

    def __maybe_convert_to_batched(self, x, CASE_3D=True, CASE_2D=False, CASE_1D=False):
        """Converts system to batched input if not batched
        (since Conv3D requires batching) and sets a flag to make clear that system
        should be converted back before output"""
        # In encoder, batched input will have dimensions 2: (M x n)
        # or 5: (M x Cin x nx x ny x nz) (for batch size M).
        # In decoder, batched input will have dimensions 2: (M x L)
        if CASE_3D:
            dims = len(x.shape)
            if dims in [2, 5]:
                self.batch = True
            elif dims in [1, 4]:
                self.batch = False
                x = x.unsqueeze(0)
            else:
                raise ValueError("AE does not accept input with dimensions {}".format(dims))

            self.batch_sz = x.shape[0]

        elif CASE_2D:
            dims = len(x.shape)
            if dims in [2, 4]:
                self.batch = True
            elif dims in [1, 3]:
                self.batch = False
                x = x.unsqueeze(0)
            else:
                raise ValueError("AE does not accept input with dimensions {}".format(dims))

            self.batch_sz = x.shape[0]
        
        elif CASE_1D:
            dims = len(x.shape)
            if dims in [3]:
                self.batch = True
            elif dims in [1,2]:
                self.batch = False
                x = x.unsqueeze(1)
            else:
                raise ValueError("AE does not accept input with dimensions {}".format(dims))

            self.batch_sz = x.shape[0]

        return x
        
    def __maybe_convert_to_non_batched(self, x, CASE_3D=True, CASE_2D=False, CASE_1D=False):
        if not self.batch and (CASE_3D or CASE_2D):
            x = x.squeeze(0)
        elif not self.batch and CASE_1D:
            x = x.squeeze(1)
        return x

    def get_list_AE_layers(self, input_size, latent_dim, hidden):
        """Helper function to get a list of the number of fc nodes or conv
        channels in an autoencoder"""
        #create a list of all dimension sizes (including input/output)
        layers = [input_size]
        #encoder:
        for size in hidden:
            layers.append(size)
        #latent representation:
        layers.append(latent_dim)
        #decoder:
        for size in hidden[::-1]: #reversed list
            layers.append(size)
        layers.append(input_size)

        return layers

    def jac_explicit(self, x):
        raise NotImplementedError("explicit Jacobian has not been implemented for this class")

