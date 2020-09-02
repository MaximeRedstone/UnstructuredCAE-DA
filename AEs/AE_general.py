""" Code from https://github.com/julianmack/Data_Assimilation with only minor modifications """

import torch.nn as nn
from collections import OrderedDict

from UnstructuredCAEDA.AEs import BaseAE
from UnstructuredCAEDA.nn.builder import NNBuilder as Build

class MODES():
    S = "SEQENTIAL"
    PA = "PARALLEL_ADD"
    PC = "PARALLEL_CONCAT"
    all = [S, PA, PC]

class GenCAE(BaseAE):
    """Initializes a general CAE using recursive data structure to define the architecture.

    Args:
        blocks (list/string)    - If string, this string describes an nn.module.
                                - If list, it is a recursive structure with shorthand
            describing the structure of nn.modules in encoder. This list will be mirrored
            and repeated (with relevant changes) in the decoder. The list must have the
            following recursive structure:
                [<MODE>, block_1, block_2, ...]
                where:
                    <MODE> (string) - operation for subsequent blocks. Must be one of:
                                     ["SEQENTIAL", "PARALLEL_ADD", "PARALLEL_CONCAT"]
                    block_x (tuple) - Tuple of length 2/3 of structure:
                        block_x = (number_x, blocks_, kwargs_block_x)
                    where:
                        number_x (int) - is number of times to repeat 'blocks_'
                        blocks_ (list/str) - of same structure as `blocks`
                        kwargs_block_x (list of dicts/dict) - optional. If list of
                                    dictionaries - each is used to init every repeated
                                    version of blocks_. Must have len(kwargs_block_x) = number_x
                                                                    If dict, all repeated
                                    versions of blocks_ will have the same kwargs dict

        activation (str)- Activation function between blocks. One of [None, "relu", "lrelu", "prelu", "GDN"]
        latent_sz (int) - latent size of fixed size input.

        # batch_norm (bool) - Whether to use batch normalization between layers (where appropriate).
        #                        Note: if a block is used that has BN included, this will override this param
        # dropout (bool) -  Whether to use dropout between all layers (where appropriate)


    """
    def __init__(self, blocks, activation = "relu", latent_sz=None, rem_final=True, dim=None):

        super(GenCAE, self).__init__()

        self.__init_activation(activation)

        self.layers_encode = self.parse_blocks(blocks, encode=True)

        self.layers_encode = self.remove_final_activation(self.layers_encode, rem_final)

        self.layers_decode = self.parse_blocks(blocks, encode=False)

        self.layers_decode = self.remove_final_activation(self.layers_decode, rem_final)

        self.latent_sz = latent_sz

        self.DIM = dim

    def setDimGenCAE(self, dim):
        self.DIM = dim

    def parse_blocks(self, blocks, encode, kwargs_ls=None):
        if isinstance(blocks, str):
            return self.parse_blocks_str(blocks, encode, kwargs_ls)
        elif isinstance(blocks, list):
            assert len(blocks) > 1, "blocks must be list of structure [<MODE>, block_1, ...]"
            mode = blocks[0]
            assert isinstance(mode, str) , "blocks[0] must be a string"
            assert mode in MODES.all

            blocks = blocks[1:] #ignore mode
            if not encode:
                blocks = blocks[::-1]
            blocks_expanded = OrderedDict()
            for idx, block in enumerate(blocks):
                assert isinstance(block, tuple)
                if len(block) == 2:
                    (num, blocks_) = block
                    kwargs_ls = None
                elif len(block) == 3:
                    (num, blocks_, kwargs_ls) = block
                    assert isinstance(blocks_, str), "Only give kwargs for node element of blocks data-structure"
                    assert isinstance(kwargs_ls, (list, dict))
                    if isinstance(kwargs_ls, list):
                        assert len(kwargs_ls) == num
                    else:
                        kwargs_ls = [kwargs_ls] * num #repeat kwargs

                    if not encode:
                        kwargs_ls = kwargs_ls[::-1] #reverse order of layers
                else:
                    raise ValueError("block must be on length 2 or 3")

                layer = OrderedDict()
                for i in range(num):
                    if kwargs_ls:
                        layers_lower = self.parse_blocks(blocks_, encode, kwargs_ls[i]) #nn.module
                    else:
                        layers_lower = self.parse_blocks(blocks_, encode)
                    layer.update({str(i): layers_lower})

                if num == 1 and mode == MODES.S:
                    layer = layer[str(0)]
                    blocks_expanded.update({str(idx): layer})
                else:
                    blocks_expanded.update({str(idx): nn.Sequential(layer)})


            if mode == MODES.S:
                if len(blocks_expanded) == 1:
                    layers_out = blocks_expanded[str(0)]
                else:
                    layers_out = nn.Sequential(blocks_expanded)
            elif mode == MODES.PA:
                raise NotImplementedError("")
                layers_out = nnParallelAdd(blocks_expanded)
            elif mode == MODES.PC:
                raise NotImplementedError("")
                layers_out = nnParallelConcat(blocks_expanded)
            else:
                raise ValueError()

            return layers_out
        else:
            raise ValueError("blocks must be of type str or list. Received type {}".format(type(blocks)))

    def parse_blocks_str(self, block, encode, layer_kwargs):

        if layer_kwargs.get("encode"):
            del layer_kwargs["encode"]

        layer_kwargs["encode"] = encode

        self.setDimGenCAE(layer_kwargs["DIM"])

        if block == "conv": #this is poorly named - simple conv?
            layer_kwargs["activation"] = self.activation
            return Build.conv(**layer_kwargs)
        elif block == "ResNeXt":
            return Build.ResNeXt(activation_fn = self.activation, **layer_kwargs)
        elif block == "resResNeXt":
            return Build.resResNeXt(activation_fn = self.activation, **layer_kwargs)
        elif block == "ResNeXt3":
            return Build.ResNeXt3(activation_fn = self.activation, **layer_kwargs)
        elif block == "RDB3":
            return Build.ResNeXtRDB3(activation_fn = self.activation, **layer_kwargs)
        elif block == "Bespoke":
            return Build.ResBespoke(activation_fn = self.activation, **layer_kwargs)
        elif block == "Tucodec":
            return Build.Tucodec(activation_fn = self.activation, **layer_kwargs)
        elif block == "Tucodec2D":
            return Build.Tucodec2D(activation_fn = self.activation, **layer_kwargs)
        elif block == "Tucodec1D":
            return Build.Tucodec1D(activation_fn = self.activation, **layer_kwargs)
        elif block == "GRDN":
            return Build.GRDN(activation_fn = self.activation, **layer_kwargs)
        elif block == "DRU":
            return Build.DRU(activation_fn = self.activation, **layer_kwargs)
        elif block == "1x1":
            Build.conv1x1(layer_kwargs)
        else:
            raise NotImplementedError("block={} is not implemented".format(block))

    def __init_activation(self, activation):
        """self.act_fn from BaseAE is depreccated in favour of initilizing
        the activation function at model init time (to give correct number
         of channels in PreLu GDN etc.)
         """

        if activation is None: #This is necessary if activation functions are included in blocks
            fn = lambda x: x #i.e. just return input
        elif activation == "lrelu":
            fn = "lrelu"
        elif activation == "relu":
            fn = "relu"
        elif activation == "prelu":
            fn = "prelu" #defer until NNBuilder()
        elif activation == "GDN":
            fn = "GDN"
        else:
            raise NotImplementedError("Activation function must be in {`relu`, `lrelu`, `prelu`, `GDN`}")
        self.activation = fn
        self.act_fn = lambda x: x #i.e. does nothing

    @staticmethod
    def remove_final_activation(module_list, rem_final):
        """Final activation from encoder and decoder must be removed after initialization"""
        if not rem_final:
            return module_list

        recursion_depth = 20
        final = module_list
        prev = module_list
        for depth in range(recursion_depth):
            try:
                save = final
                final = final[-1]
                prev = save
            except (IndexError, TypeError):
                break
        if rem_final:
            idx = -1
        else:
            idx = None


        if depth == 1:
            module_list = nn.Sequential(*list(prev.children())[:idx])
        elif depth == 2:
            module_list[-1] = nn.Sequential(*list(prev.children())[:idx])
        elif depth == 3:
            module_list[-1][-1] = nn.Sequential(*list(prev.children())[:idx])
        elif depth == 4:
            module_list[-1][-1][-1] = nn.Sequential(*list(prev.children())[:idx])
        else:
            raise NotImplementedError("Must implement activation removal to depth of {}".format(depth))

        return module_list
