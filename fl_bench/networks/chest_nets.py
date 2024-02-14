import torch
import torch.nn as nn
import os
import timm
import torchvision.models as tm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.layers import BatchNormAct2d, LayerNormAct2d, LayerNorm2d

from fl_bench.utils.misc import seed_everything
from functools import partial

model_dir = "/path/to/model_weights"
torch.hub.set_dir(model_dir)

NUM_CLASSES = 14


def resnet_50(seed):
    seed_everything(seed)
    bb = tm.resnet50()
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

resnet_50_scratch = resnet_50

def resnet_50_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.resnet50(weights="IMAGENET1K_V2")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.fc.in_features
    bb.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

resnet_50_supervised = resnet_50_pretrained_pt

def resnet_50_pretrained_ssl(seed):
    seed_everything(seed)
    model = tm.resnet50()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(
        torch.load(
            os.path.join(ssl_model_dir,
                         "resnet_50.pth"
                         )
        )
    )

    return model

resnet_50_ssl = resnet_50_pretrained_ssl


def resnet_50_pretrained_timm(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'resnet50.a1_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES)
    os.environ['https_proxy'] = ''
    return bb


def resnet_50_pretrained_dino(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50',
                        pretrained=True)
    os.environ['https_proxy'] = ''
    model.fc = nn.Linear(2048, NUM_CLASSES)
    return model

def resnet_50_pretrained_mocov3(seed):
    seed_everything(seed)
    model = tm.resnet50()
    checkpoint = torch.load(f'{model_dir}/r-50-1000ep.pth.tar', map_location="cpu")
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % 'fc'):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

# DENSENETS

def densenet_121(seed):
    seed_everything(seed)
    bb = tm.densenet121()
    num_ftrs = bb.classifier.in_features
    bb.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

densenet_121_scratch = densenet_121

def densenet_121_pretrained_pt(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = tm.densenet121(weights="IMAGENET1K_V1")
    os.environ['https_proxy'] = ''
    num_ftrs = bb.classifier.in_features
    bb.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    return bb

densenet_121_supervised = densenet_121_pretrained_pt

def densenet_121_pretrained_ssl(seed):
    seed_everything(seed)
    model = tm.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model.load_state_dict(
        torch.load(
            os.path.join(ssl_model_dir,
                         "densenet_121.pth"
                         )
        )
    )

    return model

densenet_121_ssl = densenet_121_pretrained_ssl
