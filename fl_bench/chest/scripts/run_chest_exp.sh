conda activate flare
RESULT_ROOT=/path/to/sharedscratch/miccai23/cxr14
algorithms_dir="/path/to/FL_bench/fl_bench/chest/job_configs"

seed=2024
config=federated
num_clients=4
gpu=0,0,1,1

method=fedavg
rounds=50

scheduler=onecycle
optim_resume=resume
steps_or_epochs=steps
steps=400
lr=5e-4

batch_size=64
cache_rate=0.5
grad_clipping_type=none

criterion=bce

for method in fedavg; do
    for model_name in resnet_50_supervised; do

        name="${model_name}_${method}_${grad_clipping_type}_${batch_size}_${scheduler}_${optim_resume}_${criterion}"
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