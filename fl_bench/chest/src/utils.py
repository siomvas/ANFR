import json
import os
import random
from typing import Dict
from functools import partial
from dotenv import load_dotenv
from collections import Iterable
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib import collections as matcoll

from fl_bench.networks.nf_segmentation.nf_layers import (
    ScaledWSConv2d,
    ScaledWSConvTranspose2d
)

load_dotenv()


def custom_client_datalist_json_path(datalist_json_path: str, client_id: str) -> str:
    """
    Customize datalist_json_path for each client
    Args:
         datalist_json_path: root path containing all jsons
         client_id: e.g., site-2
    """
    # Customize datalist_json_path for each client
    datalist_json_path_client = os.path.join(
        datalist_json_path,
        client_id + ".json",
    )
    return datalist_json_path_client


def nfify(module: nn.Module):
    # Replace all convolutions with ScaledWSConv2d and enable bias
    # Replace all transposed convolutions with ScaledWSTransposeConv2d
    # Replace all batchnorms with Identity
    for name, mod in module.named_children():
        target_mod = getattr(module, name)
        if isinstance(mod, torch.nn.Conv2d):
            setattr(module,
                    name,
                    ScaledWSConv2d(
                        target_mod.in_channels,
                        target_mod.out_channels,
                        target_mod.kernel_size,
                        target_mod.stride,
                        target_mod.padding,
                        target_mod.dilation,
                        target_mod.groups)
                    )
        if isinstance(mod, torch.nn.ConvTranspose2d):
            setattr(module,
                    name,
                    ScaledWSConvTranspose2d(
                        target_mod.in_channels,
                        target_mod.out_channels,
                        target_mod.kernel_size,
                        target_mod.stride,
                        target_mod.padding,
                        target_mod.dilation,
                        target_mod.groups)
                    )

        if isinstance(mod, torch.nn.BatchNorm2d):
            setattr(module, name, torch.nn.Identity())

    for name, mod in module.named_children():
        nfify(mod)


def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError('Wrong input dimensions')

    return torch.sum(x**2, dim=dim, keepdim=keepdim) ** 0.5


class AGC(torch.optim.Optimizer):
    """ Wrapper for Adaptive Gradient Clipping.
    """

    def __init__(self, params, optim, lamda=1e-2, eps=1e-3, model=None, ignore_agc=["fc"]):
        if lamda < 0.0:
            raise ValueError(f"Invalid clipping value {lamda}")
        self.optim = optim
        defaults = dict(lamda=lamda, eps=eps)
        defaults = {**defaults, **optim.defaults}

        if not isinstance(ignore_agc, Iterable):
            ignore_agc = [ignore_agc]

        if model is not None:
            assert ignore_agc not in [
                None, []], "You must specify ignore_agc for AGC to ignore fc-like(or other) layers"
            names = [name for name, module in model.named_modules()]
            for module_name in ignore_agc:
                if module_name not in names:
                    raise ModuleNotFoundError(
                        "Module name {} not found in the model".format(module_name))
            params = [{"params": list(module.parameters())} for name,
                          module in model.named_modules() if name not in ignore_agc]
        else:
            params = [{"params": params}]

        self.agc_params = params
        self.eps = torch.tensor(eps)
        self.lamda = lamda
        self.param_groups = optim.param_groups
        self.state = optim.state

        #super(AGC, self).__init__([], defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.agc_params:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_norm = torch.max(unitwise_norm(p.detach()), self.eps.to(p.device))
                grad_norm = unitwise_norm(p.grad.detach())
                max_norm = param_norm * self.lamda

                trigger = grad_norm > max_norm

                clipped_grad = p.grad * \
                    (max_norm / torch.max(grad_norm,
                                          torch.tensor(1e-6).to(grad_norm.device)))
                p.grad.detach().data.copy_(torch.where(trigger, clipped_grad, p.grad))

        return self.optim.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        for group in self.agc_params:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()


def spp_plot(name_values, *args, **kwargs):
    labels = list(name_values.keys())
    values = list(name_values.values())
    depth = np.arange(len(labels))
    plt.plot(depth, values, *args, **kwargs)
    plt.xticks(depth, labels, rotation="vertical")


def compare_name_values(name_values1, name_values2, *args, **kwargs):
    labels1 = list(name_values1.keys())
    values1 = list(name_values1.values())
    x = np.arange(len(labels1))
    y = [(values1[i], name_values2.get(labels1[i], np.nan)) for i in range(len(labels1))]

    lines = []
    for i,j in zip(x,y):
        pair = [i, j[0]], [i, j[1]]
        lines.append(pair)
    linecoll = matcoll.LineCollection(lines, colors='black')
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.add_collection(linecoll)
    ax.plot(x, values1, 'ro', label='Model1')
    values2 = [j[1] for j in y]
    ax.plot(x, values2, 'bo', label='Model2')
    plt.xticks(x, labels1, rotation="vertical")
    plt.legend()
    plt.show()


def get_average_channel_squared_mean_by_depth(model,  *args, **kwargs):
    acts = extract_activations(model, *args, **kwargs)
    values = {}
    for name, tensor in acts:
        values[name] = average_channel_squared_mean(tensor)
    return values


def get_average_channel_variance_by_depth(model,  *args, **kwargs):
    acts = extract_activations(model, *args, **kwargs)
    values = {}
    for name, tensor in acts:
        values[name] = average_channel_variance(tensor)
    return values


def average_channel_squared_mean(x):
    if x.ndim == 4:
        return (x.mean(dim=(0,2,3))**2).mean().item()
    elif x.ndim == 2:
        return (x**2).mean().item()
    else:
        raise ValueError(f"not supported shape: {x.shape}")


def average_channel_variance(x):
    if x.ndim == 4:
        return x.var(dim=(0,2,3)).mean().item()
    elif x.ndim == 2:
        return x.var(dim=0).mean().item()
    else:
        raise ValueError(f"not supported shape: {x.shape}")


def extract_activations(model, *args, **kwargs):
    acts = []
    handles = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(partial(hook, name=name, store=acts))
        handles.append(handle)
    model(*args, **kwargs)
    for handle in handles:
        handle.remove()
    return acts


def hook(self, input, output, store=None, name=None):
    if store is None:
        store = []
    store.append((name, output))


# Example usage:
# model = torchvision.models.resnet101()
# x = torch.randn(64,3,224,224)
# name_values = get_average_channel_squared_mean_by_depth(model, x)
# To match paper filter out which names to show
# paper only shows ends of res blocks
# fig = plt.figure(figsize=(15, 10))
# spp_plot(name_values)
# plt.show()
