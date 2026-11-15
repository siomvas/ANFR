"""ANFR models: registration of the CBAM variant and access to the ImageNet-1k weights.

ANFR is a normalization-free ResNet (scaled weight standardization, no activation
normalization) with a channel attention module in every block. timm already defines the
Squeeze-and-Excitation and ECA variants (``nf_seresnet50``, ``nf_ecaresnet50``) without
weights. This module adds the CBAM variant (``nf_cbamresnet50``) and resolves the
ImageNet-1k weights we pre-trained for all three.

Weights are looked up in this order:
1. ``ANFR_WEIGHTS_DIR``: a local directory holding ``<architecture>.pth`` or the original
   timm training checkpoint ``<architecture>.pth.tar``.
2. The Hugging Face Hub repository listed in ``HF_HUB_IDS``.
"""

import os

from timm.models import nfnet as _nfnet
from timm.models import register_model
from timm.models._registry import generate_default_cfgs

ARCHITECTURES = ("nf_seresnet50", "nf_ecaresnet50", "nf_cbamresnet50")

# Hub repositories of the ImageNet-1k weights, one per architecture.
HF_HUB_IDS = {
    "nf_seresnet50": "adcw447/nf_seresnet50.anfr_in1k",
    "nf_ecaresnet50": "adcw447/nf_ecaresnet50.anfr_in1k",
    "nf_cbamresnet50": "adcw447/nf_cbamresnet50.anfr_in1k",
}

if "nf_cbamresnet50" not in _nfnet.model_cfgs:
    _nfnet.model_cfgs["nf_cbamresnet50"] = _nfnet._nfres_cfg(depths=(3, 4, 6, 3), attn_layer="cbam", attn_kwargs=dict())

    default_cfgs = generate_default_cfgs({"nf_cbamresnet50": _nfnet._dcfg(url="", first_conv="stem.conv")})

    @register_model
    def nf_cbamresnet50(pretrained=False, **kwargs):
        """Normalization-Free CBAM-ResNet50."""
        return _nfnet._create_normfreenet("nf_cbamresnet50", pretrained=pretrained, **kwargs)


def pretrained_overlay(architecture):
    """Return the ``pretrained_cfg_overlay`` that points timm at the ANFR weights.

    Use as ``timm.create_model(architecture, pretrained=True,
    pretrained_cfg_overlay=pretrained_overlay(architecture))``.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"{architecture} is not an ANFR architecture: {ARCHITECTURES}")
    weights_dir = os.environ.get("ANFR_WEIGHTS_DIR")
    if weights_dir:
        for name in (f"{architecture}.pth", f"{architecture}.pth.tar"):
            path = os.path.join(weights_dir, name)
            if os.path.isfile(path):
                return dict(file=path)
        raise FileNotFoundError(
            f"ANFR_WEIGHTS_DIR is set to {weights_dir} but holds no {architecture}.pth or {architecture}.pth.tar"
        )
    hub_id = HF_HUB_IDS.get(architecture)
    if hub_id:
        return dict(hf_hub_id=hub_id)
    raise RuntimeError(
        f"No weights configured for {architecture}. Set ANFR_WEIGHTS_DIR to a directory with the checkpoint."
    )
