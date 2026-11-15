# Third-party code

## NVIDIA FLARE

Apache License 2.0, Copyright (c) NVIDIA CORPORATION. https://github.com/NVIDIA/NVFlare

- `fl_bench/utils/fedopt.py` is `nvflare/app_opt/pt/fedopt.py` of NVFlare 2.4.0, unchanged.
- The learners, trainers, validators, formatters and job configurations under `fl_bench` started from NVFlare's simulator examples and were modified for this work. Files that carry the NVIDIA copyright header keep it.
- `patches/nvflare-2.3.0-timeouts.patch` is a patch against NVFlare 2.3.0.

The text of the license is at https://www.apache.org/licenses/LICENSE-2.0.

## timm

Apache License 2.0, Copyright (c) Ross Wightman. https://github.com/huggingface/pytorch-image-models

`fl_bench/networks/anfr.py` registers a model with timm's normalization-free ResNet code. The baseline definitions in `fl_bench/networks` and `fed_het/utils/start_config.py` adapt timm model code.

## fed_het

The folder `fed_het` builds on:

- fed_het by Sara Pieri, Jose Renato Restom, Samuel Horvath and Hisham Cholakkal, the code of "Handling Data Heterogeneity via Architectural Design for Federated Visual Recognition" (NeurIPS 2023). MIT License. https://github.com/sarapieri/fed_het
- ViT-FL by Liangqiong Qu et al., on which fed_het is based. MIT License, see `fed_het/LICENSE.ViT-FL`. https://github.com/Liangqiong/ViT-FL-main
- FedConv by Peiran Xu et al., whose model definitions are used for that baseline. MIT License, see `fed_het/LICENSE.FedConv`. https://github.com/UCSC-VLAA/FedConv
