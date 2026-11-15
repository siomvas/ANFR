# fed_het: CIFAR-10 and CelebA experiments

The CIFAR-10 and CelebA experiments of the ANFR paper, including the comparisons added during review (FedConv, ViT, ConvNeXt, EfficientNet, DomainNet) and the intrinsic-dimension analysis.

This folder builds on [fed_het](https://github.com/sarapieri/fed_het) by Pieri et al., which is based on [ViT-FL](https://github.com/Liangqiong/ViT-FL-main). It keeps their training loops for FedAvg, FedProx, FedOpt and SCAFFOLD and adds the ANFR models, the baselines of the paper and the analysis scripts.

## Data

Follow the [data instructions of fed_het](https://github.com/sarapieri/fed_het#4-data--weights) and place the files under `data/`. The CIFAR-10 partitions `split_1` to `split_3` are those of ViT-FL.

## Running

```bash
python train_FedAVG.py --FL_platform ANFR-FedAVG --dataset cifar10 --split_type split_2 --seed 98
python train_SCAFFOLD.py --FL_platform ANFR-SCAFFOLD --dataset cifar10 --split_type split_2 --seed 98
python train_FedAVG.py --FL_platform ANFR-FedAVG --dataset celeba --split_type real --num_local_clients 10 --seed 999
```

`--FL_platform` selects the model by name: `ResNet`, `SE`, `NF`, `ANFR`, `ECA`, `CBAM`, `FedConv`, `ViT`, `convnext` or `Efficient`, with `scratch` for random initialization. The ANFR weights are downloaded from the Hugging Face Hub. The shell scripts are the runs of the paper and of its revisions.
