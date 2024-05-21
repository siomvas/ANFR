conda activate flare
RESULT_ROOT=/path/to/sharedscratch/miccai23/fed_isic_dp/
algorithms_dir="/path/to/FL_bench/fl_bench/fed_isic/job_configs"

config=federated_dp

seed=1
gpu=0,0,0,1,1,1

scheduler=none
steps_or_epochs=epochs
steps=1
rounds=100
lr=5e-4

batch_size=64
model_name=nf_resnet50_supervised
max_grad_norm=1.0

for lr in 1e-3 5e-4; do
    for noise_multiplier in 1.2 0.8 0.4; do
        for max_grad_norm in 1.0 0.5; do

            name="${model_name}_${method}_${batch_size}_seed_${seed}_dp_${max_grad_norm}_${noise_multiplier}_${lr}"
            target_config=${algorithms_dir}/fed_isic_${config}

            python3 /path/to/FL_bench/fl_bench/fed_isic/src/setup_dp.py \
            -job ${target_config} \
            -local_steps ${steps} \
            -num_rounds ${rounds} \
            -lr ${lr} \
            --batch_size ${batch_size} \
            --scheduler ${scheduler} \
            --steps_or_epochs ${steps_or_epochs} \
            -model_name ${model_name} \
            -seed ${seed} \
            --opt_args '{"class": "Adam"}' \
            --use_hyperion_dirs \
            --max_grad_norm ${max_grad_norm} \
            --noise_multiplier ${noise_multiplier} \
            # --amp \
        
            nvflare simulator ${target_config} -w ${RESULT_ROOT}/${name} -n 6 -gpu ${gpu}
        done
    done
done