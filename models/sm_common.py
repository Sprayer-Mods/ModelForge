# ModelForge by Sprayer Mods - GPL 3.0 license

import math
import warnings
from pkg_resources import require

from torch import (nn, zeros_like, cat, stack, is_tensor)
import torch
import torch.functional as F
import torch.nn.functional as nnF
from torch.cuda import is_available, device_count

from models.common import Conv, autopad
from utils.torch_utils import match_shape


def last_frame(x):
    """
    Transforms x (torch.tensor) from the shape:  
    [n, t, c, w, h] -> [n, c, w, h] by getting last frame in sequence (t)
    """
    return x[:, -1, ...].squeeze()

#--------------------------------------------------Temporal Modules-----------------------------------------------------
# Only apply operation on the last frame of sequence. These pair nicely with TSM module.
# These modules operate identically to the normal counterparts.

class TConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(last_frame(x))))

    def forward_fuse(self, x):
        return self.act(self.conv(last_frame(x)))

class TDWConv(TConv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class TBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        x = last_frame(x) # Only modification
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class TC3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(TBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        x = last_frame(x) # Only modification
        return self.cv3(cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class TSPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = last_frame(x)   # Only modification
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(cat((x, y1, y2, self.m(y2)), 1))

#----------------------------------------------------TSM----------------------------------------------------------------

class TemporalShift(nn.Module):
    """
    v1.1
    Note: Currently only supports a timestep of 1 (keeps a cache of prev frame)
    """
    def __init__(self, online=True):
        super(TemporalShift, self).__init__()
        self.online = online
        self.cache = None
        self.shift = self.online_shift if online else self.offline_shift

    def forward(self, x):
        return self.shift(x)

    def online_shift(self, x):
        _, c, _, _ = x.size()
        
        fold = c // 2
        if not is_tensor(self.cache):
            self.cache = x.detach().clone()

        # Note: order does matter with stack
        x = stack((self.cache.to(x.device), x), dim=0) # x.dims -> [2, n, c, w, h]
                                          # x -> [cache, x]

        out = zeros_like(x)
        #   t  n  c...
        out[1, :, :fold] = x[0, :, :fold]  # incorporate channels from cache
        out[:, :, fold:] = x[:, :, fold:]  # no shift
        
        self.cache = out[1].detach().clone() # set cache to current frame

        return out[1] # current frame after shift
    
    def offline_shift(self, x):
        _, _, c, _, _ = x.size()
        
        fold = c // 3
        self.cache = x[0, ...].detach().clone()       # Earlier frames

        x = cat((x, self.cache), dim=1)
        out = zeros_like(x)

        out[:, :-1, :fold] = x[:, 1:, :fold]# shift left
        out[:, 1:, :2 * fold] = x[:, :-1, :2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # no shift

        return out[:, 1, ...]


class TemporalPool(nn.Module):
    def __init__(self, net):
        super(TemporalPool, self).__init__()
        self.net = net

    def forward(self, x):
        x = self.temporal_pool(x)
        return self.net(x)

    @staticmethod
    def temporal_pool(x):
        n, t, c, h, w = x.size()
        # n_batch = nt // n_segment
        # x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = x.transpose(1, 2) # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        # x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        x = x.transpose(1, 2).contiguous().view(n, t // 2, c, h, w)
        return x


def make_temporal_shift(block, n_div):
    '''
    See https://github.com/open-mmlab/mmaction2/blob/master/mmaction/models/backbones/resnet_tsm.py
    Arguments:
        - block (nn.Module): The module to make into a TSM
        - n_segment (int): Number of frame segments. Default: 8.
        - n_div (int): Number of divisions for shift. Default: 8.
    '''
    return TemporalShift(
                net=block,
                n_div=n_div)


class TSMResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.out_conv = Conv(c1, c2, k, s, p, g)

        self.resblock = nn.ModuleList()
        self.resblock.append(TemporalShift(online=True))
        self.resblock.append(Conv(c1, c1, 1, 1))
        self.resblock.append(Conv(c1, c1, 3, 1))

    def forward(self, x):
        h = x
        for res in self.resblock:
            h = res(x)
        x = x + h if self.shortcut else h
        return self.out_conv(x)