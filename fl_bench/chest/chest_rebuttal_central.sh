conda activate flare
RESULT_ROOT=/path/to/sharedscratch/iclr_rebuttal/chest_central
algorithms_dir="/path/to/FL_bench/fl_bench/chest/job_configs"

config=central
gpu=0

scheduler=warmcosine
steps_or_epochs=epochs
steps=20
rounds=1
lr=5e-4

batch_size=128
cache_rate=1
optim_resume=resume
grad_clipping_type=none
grad_clipping_value=1.0
criterion=bce
fedprox_mu=0.01

for seed in 55; do
    for method in fedavg; do
        for model_name in nf_seresnet50_pretrained nf_resnet50_supervised resnet_50_supervised seresnet_50_supervised resnet50_gn_supervised; do
        # for model_name in resnet_50 nf_seresnet50 nf_resnet50 seresnet_50 resnet50_gn; do
            name="${model_name}_${method}_seed_${seed}"
            target_config=${algorithms_dir}/chest_${config}

            python3 /path/to/FL_bench/fl_bench/chest/src/setup.py \
            -job ${target_config} \
            -method ${method} \
            -num_rounds ${rounds} \
            -lr ${lr} \
            --batch_size ${batch_size} \
            --cache_rate ${cache_rate} \
            --criterion ${criterion} \
            --grad_clipping_type ${grad_clipping_type} \
            --grad_clipping_value ${grad_clipping_value} \
            --scheduler ${scheduler} \
            --fedprox_mu ${fedprox_mu} \
            --optim_resume ${optim_resume} \
            --steps_or_epochs ${steps_or_epochs} \
            -local_steps ${steps} \
            -model_name ${model_name} \
            -seed ${seed} \
            --opt_args '{"class": "Adam"}' \
            --use_hyperion_dirs \
            --amp \

            nvflare simulator ${target_config} -w ${RESULT_ROOT}/${name} -c client_All -gpu ${gpu}
        done
    done
done