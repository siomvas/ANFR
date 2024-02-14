conda activate flare
RESULT_ROOT=/path/to/sharedscratch/miccai23_chest
algorithms_dir="/path/to/FL_bench/fl_bench/chest/job_configs"

seed=2024
config=federated
num_clients=8
gpu=0,0,0,0,0,0,0,0

method=fedavg
rounds=150

scheduler=none
optim_resume=resume
steps_or_epochs=epochs
steps=1
lr=1e-3

batch_size=1
cache_rate=1.0
grad_clipping_type=none

for optim_resume in restart; do
    for model_name in unet_resnet50_dbn_pretrained unet_nf_resnet50_pretrained deeplabv3_resnet50_pretrained; do

        name="${model_name}_new_w_amp_no_clip"
        target_config=${algorithms_dir}/chest_${config}

        python3 /path/to/FL_bench/fl_bench/chest/src/setup.py \
        -job ${target_config} \
        -method ${method} \
        -num_rounds ${rounds} \
        -lr ${lr} \
        --batch_size ${batch_size} \
        --cache_rate ${cache_rate} \
        --grad_clipping_type ${grad_clipping_type} \
        --scheduler ${scheduler} \
        --optim_resume ${optim_resume} \
        --steps_or_epochs ${steps_or_epochs} \
        -local_steps ${steps} \
        -model_name ${model_name} \
        -seed ${seed} \
        --opt_args '{"class": "Adam"}' \
        --use_hyperion_dirs \
        --amp \

        nvflare simulator ${target_config} -w ${RESULT_ROOT}/${name} -n ${num_clients} -gpu ${gpu}
    done
done