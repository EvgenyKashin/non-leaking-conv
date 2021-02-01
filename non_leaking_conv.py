from typing import Union, Tuple
import sys
import functools
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def partialclass(name, cls, *args, **kwargs):
    # could I name it "Abstract Fabric" lol?
    NewCls = type(name, (cls,), {
        '__init__': functools.partialmethod(cls.__init__, *args, **kwargs)
    })

    """
    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in enviroments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).
    """
    try:
        NewCls.__module__ = sys._getframe(1).f_globals.get('__name__',
                                                           '__main__')
    except (AttributeError, ValueError):
        pass

    return NewCls


def make_tuple(x):
    if isinstance(x, int):
        x = (x, x)
    return x


class Conv2dHamming(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True):
        super(Conv2dHamming, self).__init__()

        kernel_size = make_tuple(kernel_size)
        stride = make_tuple(stride)
        padding = make_tuple(padding)
        dilation = make_tuple(dilation)

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels,
                                   kernel_size[0], kernel_size[1]))
        nn.init.kaiming_normal_(self.weight, mode='fan_out',
                                nonlinearity='relu')

        if bias is not None:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        hamming2d = np.sqrt(np.outer(np.hamming(kernel_size[0]),
                                     np.hamming(kernel_size[1])))
        hamming2d = torch.from_numpy(hamming2d).to(torch.float32)
        self.register_buffer('hamming2d', hamming2d)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def transform_weight(self):
        return self.weight * self.hamming2d

    def forward(self, input: Tensor) -> Tensor:
        out = F.conv2d(input, self.transform_weight(), self.bias, self.stride,
                       self.padding, self.dilation)
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]},"
            f" {self.weight.shape[0]}, kernel_size={self.kernel_size},"
            f" stride={self.stride}, padding={self.padding},"
            f" bias={self.bias is not None})"
        )


class Conv2dFactorized(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 ConvClass=nn.Conv2d):
        super(Conv2dFactorized, self).__init__()

        kernel_size = make_tuple(kernel_size)
        stride = make_tuple(stride)
        padding = make_tuple(padding)
        dilation = make_tuple(dilation)

        self.conv1 = ConvClass(in_channels, out_channels, (kernel_size[0], 1),
                               (stride[0], 1), (padding[0], 0),
                               (dilation[0], 1), groups, bias)
        self.conv2 = ConvClass(out_channels, out_channels, (1, kernel_size[1]),
                               (1, stride[1]), (0, padding[1]),
                               (1, dilation[1]), groups, bias)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out',
                                nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out',
                                nonlinearity='relu')

    def forward(self, input: Tensor) -> Tensor:
        x = self.conv1(input)
        return self.conv2(x)


Conv2dHamingFactorized = partialclass('Conv2dHamingFactorized',
                                      Conv2dFactorized,
                                      ConvClass=Conv2dHamming)
