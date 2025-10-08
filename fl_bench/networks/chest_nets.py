import os
import timm
import torch
import torch.nn as nn
from timm.layers import LayerNorm2d
from fl_bench.utils.misc import seed_everything

model_dir = "/path/to/model_weights"
torch.hub.set_dir(model_dir)

NUM_CLASSES = 8


def resnet_50(seed):
    seed_everything(seed)
    bb = timm.create_model(
        'resnet50.ra_in1k',
        pretrained=False,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return bb

def resnet_50_supervised(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'resnet50.ra_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    os.environ['https_proxy'] = ''
    return bb

resnet_50_pretrained = resnet_50_supervised

def seresnet_50(seed):
    seed_everything(seed)
    model = timm.create_model(
        'seresnet50.ra2_in1k',
        pretrained=False,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return model

def seresnet_50_supervised(seed):
    # Image size: train = 224 x 224, test = 288 x 288
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    seed_everything(seed)
    model = timm.create_model(
        'seresnet50.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    os.environ['https_proxy'] = ''
    return model

def nf_resnet50(seed):
    seed_everything(seed)
    model = timm.create_model(
        'nf_resnet50.ra2_in1k',
        pretrained=False,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return model

def nf_resnet50_supervised(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'nf_resnet50.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    os.environ['https_proxy'] = ''
    return model

def nf_se_resnet50(seed):
    seed_everything(seed)
    model = timm.create_model(
        'nf_seresnet50',
        pretrained=False,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return model

def caformer_supervised(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'caformer_s18.sail_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1
        )
    os.environ['https_proxy'] = ''
    return model

def resnet50_v2_supervised(seed):
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    seed_everything(seed)
    model = timm.create_model(
        'resnetv2_50.a1h_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    os.environ['https_proxy'] = ''
    return model

def resnet50_gn_supervised(seed):
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    seed_everything(seed)
    model = timm.create_model(
        'resnet50_gn.a1h_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    os.environ['https_proxy'] = ''
    return model

def resnet50_gn(seed):
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    seed_everything(seed)
    model = timm.create_model(
        'resnet50_gn.a1h_in1k',
        pretrained=False,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    os.environ['https_proxy'] = ''
    return model

def nf_seresnet50_pretrained(seed):
    model = timm.create_model(
        "nf_seresnet50",
        pretrained=True,
        pretrained_cfg_overlay=dict(file='/path/to/archive/senf_model/model_best.pth.tar'),
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return model

def nf_seresnet50(seed):
    model = timm.create_model(
        "nf_seresnet50",
        pretrained=False,
        # pretrained_cfg_overlay=dict(file='/path/to/archive/senf_model/model_best.pth.tar'),
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return model

def nf_ecaresnet50_pretrained(seed):
    model = timm.create_model(
        "nf_ecaresnet50",
        pretrained=True,
        pretrained_cfg_overlay=dict(file="/path/to/archive/ecanf_model/ecanfresnet50.pth.tar"),
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return model

def nf_cbamresnet50_pretrained(seed):
    model = timm.create_model(
        "nf_cbamresnet50",
        pretrained=True,
        pretrained_cfg_overlay=dict(file="/path/to/archive/cbamnf_model/cbamnf_resnet50.pth.tar"),
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return model

def vit_b_16_scratch(seed):
    seed_everything(seed)
    bb = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=NUM_CLASSES,
        in_chans=1)
    return bb

def vit_b_16_supervised(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1)
    return bb

def resnet_18_scratch(seed):
    seed_everything(seed)
    bb = timm.create_model(
        'resnet18',
        pretrained=False,
        num_classes=NUM_CLASSES,
        in_chans=1)
    return bb

def resnet_18_supervised(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'resnet18',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1)
    os.environ['https_proxy'] = ''
    return bb

def densenet_121_scratch(seed):
    seed_everything(seed)
    bb = timm.create_model(
        'densenet121',
        pretrained=False,
        num_classes=NUM_CLASSES,
        in_chans=1)
    return bb

def densenet_121_supervised(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'densenet121',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1)
    os.environ['https_proxy'] = ''
    return bb

def convnext_small_scratch(seed):
    bb = timm.create_model(
        'convnext_small.fb_in1k',
        pretrained=False,
        num_classes=NUM_CLASSES,
        in_chans=1)
    return bb

def convnext_small_supervised(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'convnext_small.fb_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1)
    os.environ['https_proxy'] = ''
    return bb

def efficientnet_v2_s_scratch(seed):
    bb = timm.create_model(
        'efficientnetv2_s',
        pretrained=False,
        num_classes=NUM_CLASSES,
        in_chans=1)
    return bb

def efficientnet_v2_s_supervised(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'tf_efficientnetv2_s.in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1)
    os.environ['https_proxy'] = ''
    return bb

def wide_resnet_50_2_scratch(seed):
    bb = timm.create_model(
        'wide_resnet50_2',
        pretrained=False,
        num_classes=NUM_CLASSES,
        in_chans=1)
    return bb

def wide_resnet_50_2_supervised(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'wide_resnet50_2',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1)
    os.environ['https_proxy'] = ''
    return bb

def swin_t_scratch(seed):
    bb = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=False,
        num_classes=NUM_CLASSES,
        in_chans=1)
    return bb

def swin_t_supervised(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1)
    os.environ['https_proxy'] = ''
    return bb


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


def vit(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    os.environ['https_proxy'] = ''
    return model

def dyt_vit(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    os.environ['https_proxy'] = ''
    model = convert_ln_to_dyt(model)
    return model

def dyt_convnext(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'convnext_base.fb_in1k',
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    os.environ['https_proxy'] = ''
    model = convert_ln_to_dyt(model)
    return model

def vit_pretrained(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=NUM_CLASSES
    )
    os.environ['https_proxy'] = ''
    return model

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

    def __init__(self, block, layers, num_classes=1000, in_chans=1,
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

def fedconv_normal_pretrained(seed):
    model = fedconv_normal(in_chans=1)
    pretrained_dir = '/path/to/archive/fedconv/fedconv_normal.pth.tar'
    checkpoint = torch.load(pretrained_dir, map_location='cpu')
    checkpoint = checkpoint['model']
    checkpoint['stem.0.weight'] = checkpoint['stem.0.weight'].sum(dim=1, keepdim=True)  # convert first conv to 1 channel via sum
    model.load_state_dict(checkpoint)
    model.head.fc = nn.Linear(model.head.fc.weight.shape[1], NUM_CLASSES)
    return model

def fedconv_invertup_pretrained(seed):
    model = fedconv_invertup(in_chans=1)
    pretrained_dir = '/path/to/archive/fedconv/fedconv_invertup.pth.tar'
    checkpoint = torch.load(pretrained_dir, map_location='cpu')
    checkpoint = checkpoint['model']
    checkpoint['stem.0.weight'] = checkpoint['stem.0.weight'].sum(dim=1, keepdim=True)  # convert first conv to 1 channel via sum
    model.load_state_dict(checkpoint)
    model.head.fc = nn.Linear(model.head.fc.weight.shape[1], NUM_CLASSES)
    return model

def vit_small_pretrained(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'vit_small_patch16_224',
        pretrained=True,
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    os.environ['https_proxy'] = ''
    return model