# ImageNet-1k pre-training of the ANFR models

The three ANFR-50 models were trained on ImageNet-1k with the training script of [timm](https://github.com/huggingface/pytorch-image-models) 0.9.12 at a batch size of 1,024. The published weights are the exponential moving average at the best validation epoch.

```bash
python train.py \
    --data-dir /path/to/imagenet \
    --model nf_seresnet50 \
    -b 1024 --sched step --epochs 600 --decay-epochs 1 --decay-rate .988 \
    --opt rmsproptf --opt-eps .001 --lr .09 --lr-noise 0.45 1.0 --weight-decay 7e-6 \
    --clip-mode norm --clip-grad 1.0 --drop 0.2 --drop-path 0.1 \
    --model-ema --model-ema-decay 0.9999 \
    --aa rand-m8-n2-mstd1 --mixup 0.2 --remode pixel --reprob 0.35 --recount 3 \
    --amp --pin-mem -j 4
```

Use `--model nf_ecaresnet50` for the ECA variant. For the CBAM variant, import `fl_bench.networks.anfr` in the training script so that `nf_cbamresnet50` is registered.

| Model | ImageNet-1k top-1 |
|---|---|
| `nf_seresnet50` | 80.4 |
| `nf_ecaresnet50` | 80.61 |
| `nf_cbamresnet50` | 80.37 |
