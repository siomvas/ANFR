import os
import timm
from fl_bench.utils.misc import seed_everything


NUM_CLASSES = 9


def resnet_50(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'resnet50.ra_in1k',
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    os.environ['https_proxy'] = ''
    return bb


def resnet_50_supervised(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    bb = timm.create_model(
        'resnet50.ra_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
    )
    os.environ['https_proxy'] = ''
    return bb


def seresnet_50_supervised(seed):
    # Image size: train = 224 x 224, test = 288 x 288
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    seed_everything(seed)
    model = timm.create_model(
        'seresnet50.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
    )
    os.environ['https_proxy'] = ''
    return model


def nf_resnet50_supervised(seed):
    seed_everything(seed)
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    model = timm.create_model(
        'nf_resnet50.ra2_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
    )
    os.environ['https_proxy'] = ''
    return model


def nf_se_resnet50(seed):
    seed_everything(seed)
    model = timm.create_model(
        'nf_seresnet50',
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    return model


def resnet50_gn_supervised(seed):
    os.environ['https_proxy'] = 'http://proxy.example.org:3128'
    seed_everything(seed)
    model = timm.create_model(
        'resnet50_gn.a1h_in1k',
        pretrained=True,
        num_classes=NUM_CLASSES,
    )
    os.environ['https_proxy'] = ''
    return model
