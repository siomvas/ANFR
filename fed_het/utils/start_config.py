import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import timm
from utils.anfr import pretrained_overlay
from timm.layers.norm import LayerNorm2d


# FIXUP BELOW
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class FixupBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)
        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b
        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
        out += identity
        out = self.relu(out)
        return out
class FixupBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBottleneck, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv1x1(inplanes, planes)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes, stride)
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.bias3a = nn.Parameter(torch.zeros(1))
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias3b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)
        out = self.conv2(out + self.bias2a)
        out = self.relu(out + self.bias2b)
        out = self.conv3(out + self.bias3a)
        out = out * self.scale + self.bias3b
        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
        out += identity
        out = self.relu(out)
        return out
class FixupSEBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupSEBottleneck, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv1x1(inplanes, planes)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes, stride)
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.bias3a = nn.Parameter(torch.zeros(1))
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias3b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes * self.expansion, planes // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes // 16, planes * self.expansion, kernel_size=1),
            nn.Sigmoid()
        )
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)
        out = self.conv2(out + self.bias2a)
        out = self.relu(out + self.bias2b)
        out = self.conv3(out + self.bias3a)
        out = out * self.scale + self.bias3b
        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
        out = identity + self.se(out) * out
        out = self.relu(out)
        return out
class FixupResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(FixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, FixupBottleneck):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.normal_(m.conv2.weight, mean=0, std=np.sqrt(2 / (m.conv2.weight.shape[0] * np.prod(m.conv2.weight.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.constant_(m.conv3.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x + self.bias1)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2)
        return x

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output


# Fedconv below


# FedConv

""" PyTorch FX Based Feature Extraction Helpers
Using https://pytorch.org/vision/stable/feature_extraction.html
"""
from typing import Callable, List, Dict, Union

import torch
from torch import nn

import math
from functools import partial


import torch.nn.functional as F

from timm.models._features import _get_feature_info

try:
    from torchvision.models.feature_extraction import create_feature_extractor as _create_feature_extractor
    has_fx_feature_extraction = True
except ImportError:
    has_fx_feature_extraction = False

# Layers we went to treat as leaf modules
from timm.layers import Conv2dSame, ScaledStdConv2dSame, CondConv2d, StdConv2dSame
from timm.layers.non_local_attn import BilinearAttnTransform
from timm.layers.pool2d_same import MaxPool2dSame, AvgPool2dSame

# NOTE: By default, any modules from timm.models.layers that we want to treat as leaf modules go here
# BUT modules from timm.models should use the registration mechanism below
_leaf_modules = {
    BilinearAttnTransform,  # reason: flow control t <= 1
    # Reason: get_same_padding has a max which raises a control flow error
    Conv2dSame, MaxPool2dSame, ScaledStdConv2dSame, StdConv2dSame, AvgPool2dSame,
    CondConv2d,  # reason: TypeError: F.conv2d received Proxy in groups=self.groups * B (because B = x.shape[0])
}

try:
    from timm.models.layers import InplaceAbn
    _leaf_modules.add(InplaceAbn)
except ImportError:
    pass


def register_notrace_module(module: nn.Module):
    """
    Any module not under timm.models.layers should get this decorator if we don't want to trace through it.
    """
    _leaf_modules.add(module)
    return module


# Functions we want to autowrap (treat them as leaves)
_autowrap_functions = set()


def register_notrace_function(func: Callable):
    """
    Decorator for functions which ought not to be traced through
    """
    _autowrap_functions.add(func)
    return func


def create_feature_extractor(model: nn.Module, return_nodes: Union[Dict[str, str], List[str]]):
    assert has_fx_feature_extraction, 'Please update to PyTorch 1.10+, torchvision 0.11+ for FX feature extraction'
    return _create_feature_extractor(
        model, return_nodes,
        tracer_kwargs={'leaf_modules': list(_leaf_modules), 'autowrap_functions': list(_autowrap_functions)}
    )


class FeatureGraphNet(nn.Module):
    """ A FX Graph based feature extractor that works with the model feature_info metadata
    """
    def __init__(self, model, out_indices, out_map=None):
        super().__init__()
        assert has_fx_feature_extraction, 'Please update to PyTorch 1.10+, torchvision 0.11+ for FX feature extraction'
        self.feature_info = _get_feature_info(model, out_indices)
        if out_map is not None:
            assert len(out_map) == len(out_indices)
        return_nodes = {
            info['module']: out_map[i] if out_map is not None else info['module']
            for i, info in enumerate(self.feature_info) if i in out_indices}
        self.graph_module = create_feature_extractor(model, return_nodes)

    def forward(self, x):
        return list(self.graph_module(x).values())


class GraphExtractNet(nn.Module):
    """ A standalone feature extraction wrapper that maps dict -> list or single tensor
    NOTE:
      * one can use feature_extractor directly if dictionary output is desired
      * unlike FeatureGraphNet, this is intended to be used standalone and not with model feature_info
      metadata for builtin feature extraction mode
      * create_feature_extractor can be used directly if dictionary output is acceptable
    Args:
        model: model to extract features from
        return_nodes: node names to return features from (dict or list)
        squeeze_out: if only one output, and output in list format, flatten to single tensor
    """
    def __init__(self, model, return_nodes: Union[Dict[str, str], List[str]], squeeze_out: bool = True):
        super().__init__()
        self.squeeze_out = squeeze_out
        self.graph_module = create_feature_extractor(model, return_nodes)

    def forward(self, x) -> Union[List[torch.Tensor], torch.Tensor]:
        out = list(self.graph_module(x).values())
        if self.squeeze_out and len(out) == 1:
            return out[0]
        return out


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding
    
class fedconv_block_nonnorm(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                act_layer=nn.ReLU(inplace=True), norm_layer=nn.BatchNorm2d, drop_path=None, kernel_size=3,
                layer_scale_init_value=0., drop_block=None,):
        super(fedconv_block_nonnorm, self).__init__()

        outplanes = planes * self.expansion

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        pad = get_padding(kernel_size, stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=pad,  groups=planes)  #depthwise conv

        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1)

        self.act3 = act_layer
        self.downsample = downsample
        self.drop_path = drop_path
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(outplanes)) if layer_scale_init_value > 0 else None


    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x
    

class fedconv_invert_block_nonnorm(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                act_layer=nn.ReLU(inplace=True), norm_layer=nn.BatchNorm2d, drop_path=None, kernel_size=3,
                layer_scale_init_value=0.,drop_block=None,):
        super(fedconv_invert_block_nonnorm, self).__init__()

        innerplanes = planes * self.expansion
        pad = get_padding(kernel_size, stride)

        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1)
        self.act1 = act_layer
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=kernel_size, stride=stride, padding=pad,  groups=innerplanes)
        self.conv3 = nn.Conv2d(innerplanes, planes, kernel_size=1)

        self.downsample = downsample
        self.drop_path = drop_path
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(planes)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
            
        x += shortcut

        return x



class fedconv_invertup_block_nonnorm(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                act_layer=nn.ReLU(inplace=True), norm_layer=nn.BatchNorm2d, drop_path=None, kernel_size=3,
                layer_scale_init_value=0. ,drop_block=None,
                ):
        super(fedconv_invertup_block_nonnorm, self).__init__()

        innerplanes = planes * self.expansion
        pad = get_padding(kernel_size, stride)

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, stride=stride, padding=pad,  groups=inplanes)

        self.conv2 = nn.Conv2d(inplanes, innerplanes, kernel_size=1)
        self.act2 = act_layer

        self.conv3 = nn.Conv2d(innerplanes, planes, kernel_size=1)

        self.downsample = downsample
        self.drop_path = drop_path
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(planes)) if layer_scale_init_value > 0 else None


    def forward(self, x):
        shortcut = x

        x = self.conv1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x += shortcut

        return x


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply
from timm.models.layers import DropBlock2d, DropPath, AvgPool2dSame, SelectAdaptivePool2d,create_classifier,trunc_normal_
from timm.models.registry import register_model
from collections import OrderedDict
from torch import Tensor

# __all__ = ['fedconv_normal','fedconv_invert','fedconv_invertup']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'fedconv':_cfg(interpolation='bicubic'),
}


def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


@register_notrace_module
class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def _norm(norm_layer, num_features, group_num=2):
    if norm_layer == nn.GroupNorm:
        return norm_layer(group_num, num_features)  #default: 2 group, between real ln and instance norm
    elif norm_layer==nn.LayerNorm:
        return nn.GroupNorm(1, num_features)  # same as real LN
    else:
        return norm_layer(num_features)

class UnderSp(nn.Module):
    '''
    function of softplus -value
    '''
    __constants__ = ['beta', 'threshold']
    beta: int
    threshold: int

    def __init__(self, beta: int = 1, threshold: int = 20, value: float = 1.) -> None:
        super(UnderSp, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return F.softplus(input, self.beta, self.threshold)- self.value

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    #p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
        _norm(norm_layer, out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        _norm(norm_layer, out_channels)
    ])


def drop_blocks(drop_block_rate=0.):
    return [
        None, None,
        DropBlock2d(drop_block_rate, 5, 0.25) if drop_block_rate else None,
        DropBlock2d(drop_block_rate, 3, 1.00) if drop_block_rate else None]


def make_blocks(
        block_fn, channels, block_repeats, inplanes, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., stride_stage=1,
        invert=False, **kwargs):
    stages = []

    net_num_blocks = sum(block_repeats)
    net_block_idx = 0

    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx < stride_stage else 2

        ds_out = planes * block_fn.expansion if not invert else planes
        if stride != 1 or inplanes != ds_out:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=ds_out, kernel_size=down_kernel_size,
                stride=stride, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)
        else:
            downsample = None

        block_kwargs = dict(drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule

            blocks.append(block_fn(
                inplanes, planes, stride, downsample,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            
            inplanes = planes * block_fn.expansion  if not invert else planes
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))

    return stages


class Fedconv(nn.Module):

    def __init__(self, block, layers, num_classes=1000, in_chans=3,
                 output_stride=32, down_kernel_size=1, avg_down=False,
                 act_layer=nn.GELU(), norm_layer=LayerNorm2d, 
                 drop_rate=0.0, drop_path_rate=0.,
                 drop_block_rate=0., global_pool='avg', block_args=None,
                 channels = [64, 128, 256, 512], stride_stage =1, 
                 kernel_size=3, invert=False, 
                 layer_scale_init_value=1e-6 ):

        block_args = block_args or dict()
        assert output_stride == 32

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(Fedconv, self).__init__()

        # Stem
        inplanes = channels[0]
        
        self.stem = nn.Sequential(
        nn.Conv2d(in_chans, inplanes//2, kernel_size=3, stride=2, padding=1, bias=True),
        _norm(norm_layer, inplanes//2),
        nn.Conv2d(inplanes//2, inplanes, kernel_size=3, stride=2, padding=1, bias=True),
        _norm(norm_layer, inplanes),
        nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=True),
        )
        
        stage_modules = make_blocks(
            block, channels, layers, inplanes,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, stride_stage=stride_stage, 
            kernel_size=kernel_size, invert=invert, 
            layer_scale_init_value=layer_scale_init_value, **block_args)

        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        

        # Head (Pooling and Classifier)
        self.num_features = channels[-1] * block.expansion if not invert else channels[-1]
        self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', _norm(norm_layer, self.num_features) if norm_layer!=nn.InstanceNorm2d else nn.Identity()),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('drop', nn.Dropout(self.drop_rate)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
            ]))

        named_apply(partial(_init_weights), self)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool='avg'):
            # pool -> norm -> fc
        self.head = nn.Sequential(OrderedDict([
            ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
            ('norm', self.head.norm),
            ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
            ('drop', nn.Dropout(self.drop_rate)),
            ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        ]))

    def forward_features(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.constant_(module.bias, 0)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        Fedconv, variant, pretrained,
        # model_cfg=default_cfgs[variant],
        **kwargs)
    


def fedconv_normal(pretrained=False, **kwargs):  
    '''
    FedConv-Normal
    '''
    model_args = dict(block=fedconv_block_nonnorm, layers=[3, 3, 5, 3], channels=[96, 192, 384, 768], 
                       act_layer=nn.SiLU(), kernel_size=9, 
                       **kwargs)
    return _create_resnet('fedconv', pretrained, **model_args)


def fedconv_invert(pretrained=False, **kwargs): 
    '''
    FedConv-Invert
    '''
    model_args = dict(block=fedconv_invert_block_nonnorm, layers=[3, 3, 6, 3], channels=[96, 192, 384, 768], 
                       act_layer=nn.SiLU(), kernel_size=9, 
                      invert=True, **kwargs)
    return _create_resnet('fedconv', pretrained, **model_args)



def fedconv_invertup(pretrained=False, **kwargs):  
    '''
    FedConv-InvertUp
    '''
    cfg = dict(block=fedconv_invertup_block_nonnorm, layers=[3, 4, 9, 3], channels=[96, 192, 384, 768], 
                       act_layer=nn.SiLU(), kernel_size=9, 
                      invert=True, **kwargs)
    return _create_resnet('fedconv', pretrained, **cfg)

def fedconv_normal_pretrained(seed, NUM_CLASSES=10):
    model = fedconv_normal()
    pretrained_dir = os.path.join(os.environ.get("FEDCONV_WEIGHTS_DIR", "weights/fedconv"), "fedconv_normal.pth.tar")
    checkpoint = torch.load(pretrained_dir, map_location='cpu')
    checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)
    model.head.fc = nn.Linear(model.head.fc.weight.shape[1], NUM_CLASSES)
    return model

def fedconv_invertup_pretrained(seed, NUM_CLASSES=10):
    model = fedconv_invertup()
    pretrained_dir = os.path.join(os.environ.get("FEDCONV_WEIGHTS_DIR", "weights/fedconv"), "fedconv_invertup.pth.tar")
    checkpoint = torch.load(pretrained_dir, map_location='cpu')
    checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)
    model.head.fc = nn.Linear(model.head.fc.weight.shape[1], NUM_CLASSES)
    return model

def fedconv_invertup_scratch(seed, NUM_CLASSES=10):
    model = fedconv_invertup()
    # pretrained_dir = os.path.join(os.environ.get("FEDCONV_WEIGHTS_DIR", "weights/fedconv"), "fedconv_invertup.pth.tar")
    # checkpoint = torch.load(pretrained_dir, map_location='cpu')
    # checkpoint = checkpoint['model']
    # model.load_state_dict(checkpoint)
    model.head.fc = nn.Linear(model.head.fc.weight.shape[1], NUM_CLASSES)
    return model

def efficientnet_v2_s_supervised(seed, NUM_CLASSES=10):
    bb = timm.create_model(
        'tf_efficientnetv2_s.in1k',
        pretrained=True,
        num_classes=NUM_CLASSES)
    return bb


def print_options(args, model):
    message = ''

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = num_params / 1000000

    message += "================ FL train of %s with total model parameters: %2.1fM  ================\n" % (args.FL_platform, num_params)

    message += '++++++++++++++++ Other Train related parameters ++++++++++++++++ \n'

    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '++++++++++++++++  End of show parameters ++++++++++++++++ '


    ## save to disk of current log
    args.file_name = os.path.join(args.output_dir, 'log_file.txt')

    with open(args.file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')

    print(message)


def initization_configure(args, vis= False):

    args.device = torch.device("cuda:{gpu_id}".format(gpu_id = args.gpu_ids) if torch.cuda.is_available() else "cpu")

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not args.device == 'cpu':
        torch.cuda.manual_seed(args.seed)

    # select class number 
    dataset_class_map = {
        'cifar10': 10,
        'pacs': 7,
        'gldk23': 203,
        'isic19': 8,
        'celeba': 2,
        'domainnet': 10
    }
    args.num_classes = dataset_class_map[args.dataset]


    # select and initialize model 
    if "ResNet" in args.FL_platform:

        if args.norm and 'GN' in args.norm: 

            if "scratch" not in args.FL_platform:
                print('Architecture: ResNet-50 with GN')
                model = timm.create_model('resnet50_gn.a1h_in1k', pretrained=True, num_classes=args.num_classes)
            else:
                print('Architecture: Untrained ResNet-50 with GN')
                model = timm.create_model('resnet50_gn.a1h_in1k', pretrained=False, num_classes=args.num_classes)                

        else:
            
            if "scratch" not in args.FL_platform:
                print('Architecture: ResNet-50')
                model = timm.create_model('resnet50.ra_in1k',pretrained=True, num_classes=args.num_classes)
            else:
                print('Architecture: Untrained ResNet-50')
                model = timm.create_model('resnet50', pretrained=False, num_classes=args.num_classes)

        # model.fc = nn.Linear(model.fc.weight.shape[1], args.num_classes)
        model.to(args.device)
    elif "ECA" in args.FL_platform:
        print("Architecture: NF-ECA-ResNet-50")
        model = timm.create_model(
            'nf_ecaresnet50', pretrained=True,
            pretrained_cfg_overlay=pretrained_overlay("nf_ecaresnet50"),
            num_classes=args.num_classes)
        model.to(args.device)
        
    elif "CBAM" in args.FL_platform:
        print("Architecture: NF-CBAM-ResNet-50")
        model = timm.create_model(
            'nf_cbamresnet50', pretrained=True,
            pretrained_cfg_overlay=pretrained_overlay("nf_cbamresnet50"),
            num_classes=args.num_classes)
        model.to(args.device)
    
    elif "DyT_ViT" in args.FL_platform:
        if 'scratch' in args.FL_platform:
            print("Architecture: Untrained DyT_ViT")
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=args.num_classes)
        else:
            print("Architecture: DyT_ViT")
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=args.num_classes)
        model = convert_ln_to_dyt(model)
        model.to(args.device)
    
    elif "Fixup" in args.FL_platform:
        if 'scratch' in args.FL_platform:
            if "FixupSE" in args.FL_platform:
                print('Architecture: Untrained FixupSE-ResNet-50')
                model = FixupResNet(FixupSEBottleneck, [3, 4, 6, 3], num_classes=args.num_classes)
            else:
                print('Architecture: Untrained Fixup-ResNet-50')
                model = FixupResNet(FixupBottleneck, [3, 4, 6, 3], num_classes=args.num_classes)
        else:
            raise ValueError("Fixup model should be trained from scratch")
        model.to(args.device)

    elif "ViT" in args.FL_platform:
        if 'scratch' in args.FL_platform:
            print("Architecture: Untrained ViT_B_16")
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=args.num_classes)
        else:
            print("Architecture: ViT_S_16")
            model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=args.num_classes)
        model.to(args.device)
    
    elif "convnext" in args.FL_platform:
        print("Architecture: ConvNeXt Small")
        model = timm.create_model('convnext_small.fb_in1k', pretrained=True, num_classes=args.num_classes)
        model.to(args.device)

    elif "SE" in args.FL_platform:
        if 'scratch' in args.FL_platform:
            print('Architecture: Untrained SE-ResNet-50')
            model = timm.create_model('seresnet50', pretrained=False, num_classes=args.num_classes)
            model.to(args.device)
        else:
            print('Architecture: SE-ResNet-50')
            model = timm.create_model('seresnet50.ra2_in1k', pretrained=True, num_classes=args.num_classes)
            model.to(args.device)

    elif "ANFR" in args.FL_platform:
        if 'scratch' in args.FL_platform:
            print('Architecture: Untrained ANFR-ResNet-50')
            model = timm.create_model('nf_seresnet50', pretrained=False, num_classes=args.num_classes)
            model.to(args.device)
        else:
            print('Architecture: ANFR-ResNet-50')
            model = timm.create_model(
                'nf_seresnet50', pretrained=True,
                pretrained_cfg_overlay=pretrained_overlay("nf_seresnet50"),
                num_classes=args.num_classes)
            model.to(args.device)
    
    elif "FedConv" in args.FL_platform:
        if 'scratch' in args.FL_platform:
            print('Architecture: Untrained FedConv-InvertUp')
            model = fedconv_invertup_scratch(args.seed, args.num_classes)
        else:
            print('Architecture: FedConv-InvertUp')
            model = fedconv_invertup_pretrained(args.seed, args.num_classes)
        model.to(args.device)
    
    elif "Efficient" in args.FL_platform:
        print('Architecture: EfficientNet V2 Small')
        model = efficientnet_v2_s_supervised(0)
        model.to(args.device)

    elif "NF" in args.FL_platform:
        if 'scratch' in args.FL_platform:
            print('Architecture: Untrained NF-ResNet-50')
            model = timm.create_model('nf_resnet50', pretrained=False, num_classes=args.num_classes)
            model.to(args.device)
        else:
            print('Architecture: NF-ResNet-50')
            model = timm.create_model('nf_resnet50.ra2_in1k', pretrained=True, num_classes=args.num_classes)
            model.to(args.device)

        
    name_parts = [
        args.FL_platform,
        args.dataset,
        args.split_type,
        f"run_{args.n}",
        f"lr_{args.learning_rate}",
        f"Seed_{args.seed}"
    ]

    if args.norm:
        name_parts.insert(1, args.norm)

    args.name_run = '_'.join(name_parts)


    args.output_dir = os.path.join('output', args.FL_platform, args.dataset, args.name_run)
    os.makedirs(args.output_dir, exist_ok=True)
    print_options(args, model)

    # set train val related paramteres
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}
    args.current_test_acc_avg = {}

    return model
