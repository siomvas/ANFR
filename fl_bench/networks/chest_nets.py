import os

import timm
import torch

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


def nf_seresnet50_pretrained(seed):
    model = timm.create_model(
        "nf_seresnet50",
        pretrained=True,
        pretrained_cfg_overlay=dict(file='/path/to/archive/senf_model/model_best.pth.tar'),
        num_classes=NUM_CLASSES,
        in_chans=1
    )
    return model
