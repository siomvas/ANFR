conda activate flare
RESULT_ROOT="/path/to/sharedscratch/tmlr_rebuttal"
algorithms_dir="/path/to/FL_bench/fl_bench/fed_isic/job_configs"

config=federated

seed=24
gpu=0,0,0,0,0,0

scheduler=onecycle
optim_resume=resume
steps_or_epochs=steps
steps=200
rounds=80
lr=5e-4
fedprox_mu=0.00

grad_clipping_type=none
grad_clipping_value=1.0
batch_size=64

for seed in 999; do
    for method in fedavg; do
        for model_name in convnext_small_pretrained_pt; do

            name="${model_name}_${method}_rebuttal_seed_${seed}"
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
            --amp \
            
            nvflare simulator ${target_config} -w ${RESULT_ROOT}/${name} -n 6 -gpu ${gpu}
        done
    done
done
