conda activate flare
RESULT_ROOT=/path/to/sharedscratch/miccai23/chest
algorithms_dir="/path/to/FL_bench/fl_bench/chest/job_configs"

seed=2024
config=federated_copy
clients=client_cxr14,client_padchest,client_cxp_young,client_cxp_old
gpu=0,0,0,0

scheduler=onecycle
optim_resume=resume
steps_or_epochs=steps
steps=500
rounds=40
lr=1e-3

batch_size=64
cache_rate=1.0
grad_clipping_type=none

criterion=bce


for lr in 1e-3 1e-4; do
    for grad_clipping_type in none norm; do
        for criterion in fl; do
            for method in fedopt; do
                for model_name in resnet_50_supervised nf_resnet50_supervised; do

                    name="${model_name}_${method}_${lr}_${grad_clipping_type}_${rounds}_${optim_resume}_${criterion}"
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

                    nvflare simulator ${target_config} -w ${RESULT_ROOT}/${name} -c ${clients} -gpu ${gpu}
                done
            done
        done
    done
done