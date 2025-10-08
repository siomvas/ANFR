conda activate flare
RESULT_ROOT="/path/to/sharedscratch/icml_rebuttal"
algorithms_dir="/path/to/FL_bench/fl_bench/fed_isic/job_configs"

config=federated_rebuttal

gpu=0,0,0,0,0,0

scheduler=onecycle
optim_resume=resume
steps_or_epochs=steps
steps=400
rounds=80
lr=5e-4
fedprox_mu=0.00
grad_clipping_type=none
grad_clipping_value=0.00
batch_size=64

for seed in 99; do
    for method in fedavg; do
        # for model_name in resnet_50_pretrained_timm nf_resnet50_supervised seresnet_50_supervised resnet50_gn_supervised nf_seresnet50_pretrained; do
        # for model_name in resnet_50 nf_seresnet50_pretrained nf_resnet50 seresnet_50 resnet50_gn; do
        for model_name in fixup_seresnet50; do
            name="${model_name}_${method}_seed_${seed}_fix"
            target_config=${algorithms_dir}/fed_isic_${config}

            python3 /path/to/FL_bench/fl_bench/fed_isic/src/setup.py \
                -job ${target_config} \
                -method ${method} \
                -local_steps ${steps} \
                -num_rounds ${rounds} \
                -lr ${lr} \
                --fedprox_mu ${fedprox_mu} \
                --batch_size ${batch_size} \
                --grad_clipping_type ${grad_clipping_type} \
                --grad_clipping_value ${grad_clipping_value} \
                --scheduler ${scheduler} \
                --optim_resume ${optim_resume} \
                --steps_or_epochs ${steps_or_epochs} \
                -model_name ${model_name} \
                -seed ${seed} \
                --opt_args '{"class": "Adam"}' \
                --use_hyperion_dirs \
                --amp

            nvflare simulator ${target_config} -w ${RESULT_ROOT}/${name} -n 6 -gpu ${gpu}
        done
    done
done
