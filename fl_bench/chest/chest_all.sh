conda activate flare
RESULT_ROOT=/path/to/sharedscratch/miccai23/chest
algorithms_dir="/path/to/FL_bench/fl_bench/chest/job_configs"

seed=2024
config=federated
clients=client_cxr14,client_padchest,client_cxp_young,client_cxp_old
gpu=0,0,0,0

scheduler=warmcosine
steps_or_epochs=steps
steps=200
rounds=20
lr=5e-4

batch_size=128
cache_rate=1.0
optim_resume=resume
grad_clipping_type=none
criterion=bce
fedprox_mu=0.01

for seed in 1; do
    for method in fedprox scaffold; do
        for model_name in resnet_50_supervised nf_resnet50_supervised seresnet_50_supervised nf_seresnet50_pretrained resnet50_gn_supervised; do

            name="${model_name}_${method}_${steps}_${steps_or_epochs}_${lr}_${scheduler}_bs_${batch_size}_strong_aug_mindecay_short_final_seed_${seed}"
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
            --scheduler ${scheduler} \
            --fedprox_mu ${fedprox_mu} \
            --optim_resume ${optim_resume} \
            --steps_or_epochs ${steps_or_epochs} \
            -local_steps ${steps} \
            -model_name ${model_name} \
            -seed ${seed} \
            --opt_args '{"class": "Adam"}' \
            --use_hyperion_dirs \
            # --amp \

            nvflare simulator ${target_config} -w ${RESULT_ROOT}/${name} -c ${clients} -gpu ${gpu}
        done
    done
done