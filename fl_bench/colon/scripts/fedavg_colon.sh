conda activate flare
RESULT_ROOT=/path/to/sharedscratch/miccai23/colon
algorithms_dir="/path/to/FL_bench/fl_bench/colon/job_configs"

config=federated

seed=24
gpu=0,0,0
num_clients=3
scheduler=none
optim_resume=resume
steps_or_epochs=epochs
steps=1
rounds=20
lr=3e-4

grad_clipping_type=none
grad_clipping_value=1.0
batch_size=64

for seed in 1; do
    for method in fedavg; do
        for batch_size in 64; do
            for model_name in resnet_50_supervised nf_resnet50_supervised; do

                name="${model_name}_${method}_${lr}_${batch_size}_${scheduler}_seed_${seed}"
                target_config=${algorithms_dir}/colon_${config}

                python3 /path/to/FL_bench/fl_bench/colon/src/setup.py \
                -job ${target_config} \
                -method ${method} \
                -local_steps ${steps} \
                -num_rounds ${rounds} \
                -lr ${lr} \
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
                --amp \
                
                nvflare simulator ${target_config} -w ${RESULT_ROOT}/${name} -n ${num_clients} -gpu ${gpu}
            done
        done
    done
done