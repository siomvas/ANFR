# ANFR

Code and pre-trained models for **An Architecture Built for Federated Learning: Addressing Data Heterogeneity through Adaptive Normalization-Free Feature Recalibration** (TMLR, 2025).

[Paper (OpenReview)](https://openreview.net/forum?id=GtdYFLsblb) | [arXiv](https://arxiv.org/abs/2410.02006) | [Models on Hugging Face](https://huggingface.co/adcw447)

Adaptive Normalization-free Feature Recalibration (ANFR) addresses data heterogeneity in the backbone model. It replaces activation normalization with weight standardization and adds channel attention to every block, so that the statistics the attention sees are consistent across clients. ANFR works with any aggregation method, in global and personalized federated learning, and under differential privacy.

## Pre-trained models

ANFR-50 with three attention modules, pre-trained on ImageNet-1k. These are the weights the experiments of the paper start from.

| Model | Attention | Parameters | ImageNet-1k top-1 | Weights |
|---|---|---|---|---|
| `nf_seresnet50` | Squeeze-and-Excitation | 28.09 M | 80.4 | [adcw447/nf_seresnet50.anfr_in1k](https://huggingface.co/adcw447/nf_seresnet50.anfr_in1k) |
| `nf_ecaresnet50` | ECA | 25.56 M | 80.61 | [adcw447/nf_ecaresnet50.anfr_in1k](https://huggingface.co/adcw447/nf_ecaresnet50.anfr_in1k) |
| `nf_cbamresnet50` | CBAM | 28.07 M | 80.37 | [adcw447/nf_cbamresnet50.anfr_in1k](https://huggingface.co/adcw447/nf_cbamresnet50.anfr_in1k) |

```python
import timm

model = timm.create_model("hf_hub:adcw447/nf_seresnet50.anfr_in1k", pretrained=True, num_classes=8)
```

The SE and ECA architectures are part of timm. The CBAM variant is registered by `fl_bench/networks/anfr.py`, so import that module before creating it. The ImageNet training command is in `pretraining/README.md`.

## What is here

| Path | Content |
|---|---|
| `fl_bench/networks/anfr.py` | The ANFR models: registration of the CBAM variant and loading of the weights |
| `fl_bench/fed_isic` | Fed-ISIC2019, including the differentially private and intrinsic-dimension variants |
| `fl_bench/chest` | FedChest, with `src/chest_preparation.py`, the script that built the dataset for the paper |
| `fl_bench/colon` | FedPathology, with its Dirichlet partitions and differentially private training |
| `fl_bench/networks` | Model factories for ANFR and the baselines (ResNet-50 with batch or group normalization, SE-ResNet-50, NF-ResNet-50) |
| `fl_bench/utils` | Shared helpers, cross-validation, reporting and the FedOpt generator |
| `fed_het` | The CIFAR-10 and CelebA experiments |
| `pretraining` | The ImageNet-1k recipe of the ANFR weights |
| `patches` | An optional patch that raises NVFlare's simulator time-outs |

The medical experiments run in the [NVIDIA FLARE](https://github.com/NVIDIA/NVFlare) simulator with FedAvg, FedProx, FedOpt, SCAFFOLD, FedBN and FedPer.

## Setup

The experiments ran with Python 3.8, PyTorch 2.1, timm 0.9.12 and NVFlare 2.3.0.

```bash
git clone https://github.com/siomvas/ANFR.git && cd ANFR
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

FedOpt needs no extra step. The experiments used the FedOpt generator that NVFlare released in 2.4.0, which handles models with batch normalization correctly, and that file ships here as `fl_bench/utils/fedopt.py`.

If clients time out during long local training, apply the time-out patch to your NVFlare install:

```bash
cd $(python -c "import nvflare, os; print(os.path.dirname(os.path.dirname(nvflare.__file__)))")
patch -p1 < /path/to/ANFR/patches/nvflare-2.3.0-timeouts.patch
```

## Data

Copy `.env.example` to `.env` and set the folders.

| Variable | What it points to |
|---|---|
| `FEDISIC_ROOT` | Fed-ISIC2019, preprocessed as in [FLamby](https://github.com/owkin/FLamby) |
| `CHEST_ROOT` | FedChest: the images and the `datalists` folder. See [siomvas/FedChest](https://github.com/siomvas/FedChest) |
| `COLON_ROOT` | FedPathology: `nct_wsi_100k_no_norm.npz`, packed from NCT-CRC-HE-100K-NONORM and CRC-VAL-HE-7K (Kather et al.) |
| `ANFR_WEIGHTS_DIR` | Optional. A folder with the ANFR checkpoints, used in place of the Hugging Face Hub |
| `FEDCONV_WEIGHTS_DIR` | Optional. FedConv checkpoints, for that baseline only |

## Running

Each run has two steps. `setup.py` writes the choices into a job configuration, and the NVFlare simulator executes it.

```bash
python fl_bench/chest/src/setup.py -job fl_bench/chest/job_configs/chest_federated -method fedavg \
    -model_name nf_seresnet50_pretrained -num_rounds 20 -local_steps 200 -lr 5e-4 --batch_size 128 \
    --scheduler warmcosine --criterion bce --steps_or_epochs steps --optim_resume resume \
    --opt_args '{"class": "Adam"}' -seed 2024 --amp
nvflare simulator fl_bench/chest/job_configs/chest_federated -w /path/to/results/run_name \
    -c client_cxr14,client_padchest,client_cxp_young,client_cxp_old -gpu 0,0,0,0
```

The scripts under each dataset's `scripts` folder are the ones that produced the paper's tables, kept as they ran with `/path/to` in place of the folders of our cluster.

### CIFAR-10 and CelebA

These experiments use `fed_het`, a separate code base without NVFlare that builds on the code of Pieri et al. See `fed_het/README.md`.

```bash
cd fed_het
python train_FedAVG.py --FL_platform ANFR-FedAVG --dataset cifar10 --split_type split_2 --seed 98
```

## Citation

```bibtex
@article{siomos2025anfr,
  title   = {An Architecture Built for Federated Learning: Addressing Data Heterogeneity through Adaptive Normalization-Free Feature Recalibration},
  author  = {Siomos, Vasilis and Passerat-Palmbach, Jonathan and Tarroni, Giacomo},
  journal = {Transactions on Machine Learning Research},
  year    = {2025},
  url     = {https://openreview.net/forum?id=GtdYFLsblb}
}
```

## License

MIT, see `LICENSE.md`. Third-party code and its licenses are listed in `NOTICE.md`.

Related work from the same thesis: [ARIA](https://github.com/siomvas/ARIA), [FedCLAM](https://github.com/siomvas/FedCLAM) and [FDS](https://github.com/siomvas/fds).
