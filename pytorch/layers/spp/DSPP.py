# coding=utf8
# --------------------------------------------------------
# Dilated Spatial Pyraid Pooling Layer Using Pytorch v1.0
# Licensed under The MIT License
# Written by Ruiyuan Lu
# 2019-02-18
# --------------------------------------------------------

import torch
import numbers
from math import ceil, floor
from torch.nn import functional as F

class DSPP2d(torch.nn.Module):
    r"""
    Dilated Spatial Pyramid Pooling Layer of 2D tensors.
    Equivalent to Spatial Pyramid Pooling if dilation = 1.

    SPP: 《Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition》
    https://arxiv.org/abs/1406.4729

    .. math::

        Window_Size = \left\lceil\frac{Size_{in}}{Size_{out}}\right\rceil \\
        Stride = \left\lfloor\frac{Size_{in}}{Size_{out}}\right\rfloor \\
        Padding = \left\lceil \frac{(Size_{out}-1)\times Sride + Dilation \times
            (K-1) - Size_{in} + 1}{2} \right\rceil \\

    Args:
        output_size (int, tuple of ints, 2d tuple of ints): The output size
            of multi-level SPP pooling. The level of Dilated Spatial Pyramid 
            Pooling is defined by the length of output_size.
            E.g. (4, 2, 1) == ((4,4), (2,2), (1,1)) of 3 level SPP pooling.
        dilation (int, tuple of ints, 2d tuple of ints): The dilation of multi-level SPP pooling.
            E.g. (3, 2, 1) == ((3,3), (2,2), (1,1)) of 3 level dilated SPP pooling.
            Note: If the dilation is a iterable obj of numbers (list, tuple, etc),
            len(dilation) == len(output_size) required, for the DSPP level is defined by len(output_size).
        pool (str): Pooling method of each Spatial Pyramid step. One of ('max', 'avg').
    """

    def __init__(self, output_size=(4, 2, 1), dilation=1, pool='max'):
        super(DSPP2d, self).__init__()
        self._level = 1 if isinstance(output_size, numbers.Real) else len(output_size) # dspp level
        self._output_size = self._to_tuple_2d(output_size, 'output_size', 1)
        self._dilation = self._to_tuple_2d(dilation, 'dilation', 1)
        self.pool = pool

    def forward(self, x):
        r"""Dilated SPP pooling"""
        N, C, H, W = x.shape # channel-first
        vecs = []
        for s, d in zip(self.output_size, self.dilation):
            size = ceil(H / s[0]), ceil(W / s[1]) # window size
            stride = floor(H / s[0]), floor(W / s[1]) # stride
            pad_h = ceil(((s[0] - 1) * stride[0] + d[0] * (size[0] - 1) - H + 1) / 2)
            pad_w = ceil(((s[1] - 1) * stride[1] + d[1] * (size[1] - 1) - W + 1) / 2)
            padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h)) # zero padding
            dspp = self._pool(padded, size, stride, padding=0, dilation=d) # pooling
            vecs.append(dspp.view(N, -1)) # flatten
        return torch.cat(vecs, dim=1) # aggregation

    @property
    def output_size(self):
        return self._output_size

    @property
    def level(self):
        return self._level

    @property
    def dilation(self):
        return self._dilation

    @property
    def pool(self):
        return 'max_2d' if self._pool == F.max_pool2d else 'avg_2d'

    @pool.setter
    def pool(self, method):
        method = method.lower()
        if method =='max':
            self._pool = F.max_pool2d
        elif method == 'avg':
            self._pool = F.avg_pool2d
        else:
            raise ValueError(f"only support ('max'|'avg') pooling, '{method}' found.")

    def _to_tuple_2d(self, param, param_name, min_val):
        """Check and transform params to tuple 2D format."""
        if isinstance(param, numbers.Real):
            if param < min_val:
                raise ValueError(f"'{param_name}' >= {min_val} required.")
            else:
                return ((param, param),) * self.level
        elif param:
            if len(param) != self.level:
                raise ValueError(f"SPP level: '{self.level}' != {param_name} level: '{len(param)}'")
            final_params = []
            for v in param:
                if isinstance(v, numbers.Real):
                    if v < min_val:
                        raise ValueError(f"'{param_name}' >= {min_val} required.")
                    else:
                        final_params.append((v, v))
                elif (len(v) == 2 and
                      isinstance(v[0], numbers.Real) and
                      isinstance(v[1], numbers.Real)):
                        if v[0] >= min_val and v[1] >= min_val:
                            final_params.append(tuple(v))
                        else:
                            raise ValueError(f"'{param_name}' >= {min_val} required.")
                else:
                    raise TypeError(f"Invalid type '{type(v)}' of element '{v}' found in " +
                                    f"'{param_name}'. (int|float) only.")
            return tuple(final_params)
        else:
            raise TypeError(f"Invalid param: '{param_name}' = '{param}'. Numbers or iterable obj of numbers required.")

    def __repr__(self):
        return (self.__class__.__name__ + f"(level={self.level}, output_size=" +
            f"{self.output_size}), dilation={self.dilation}, pool={self.pool}.")
