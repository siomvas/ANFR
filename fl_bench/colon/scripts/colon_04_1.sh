conda activate flare
RESULT_ROOT=/path/to/sharedscratch/miccai23/colon_new
algorithms_dir="/path/to/FL_bench/fl_bench/colon/job_configs"

config=federated_04_1
gpu=0,0,0
num_clients=3
scheduler=none
optim_resume=resume
steps_or_epochs=steps
fedprox_mu=0.0
steps=250
rounds=25
lr=1e-4
method=fedavg
grad_clipping_type=agc
grad_clipping_value=0.04
batch_size=128
opt_args='{"class": "AdamW", "weight_decay": 1e-2}'

for seed in 1; do
    for method in fedavg; do
        for model_name in resnet_50_supervised nf_resnet50_supervised resnet50_gn_supervised seresnet_50_supervised nf_seresnet50_pretrained; do

            name="${model_name}_${method}_scheduler_${scheduler}_lr_${lr}_batch_${batch_size}_grad_${grad_clipping_type}_${grad_clipping_value}_seed_${seed}"
            target_config=${algorithms_dir}/colon_${config}

            python3 /path/to/FL_bench/fl_bench/colon/src/setup.py \
            -job ${target_config} \
            -method ${method} \
            -alpha 0.5 \
            -local_steps ${steps} \
            -num_rounds ${rounds} \
            -lr ${lr} \
            --batch_size ${batch_size} \
            --grad_clipping_type ${grad_clipping_type} \
            --grad_clipping_value ${grad_clipping_value} \
            --scheduler ${scheduler} \
            --optim_resume ${optim_resume} \
            --fedprox_mu ${fedprox_mu} \
            --steps_or_epochs ${steps_or_epochs} \
            -model_name ${model_name} \
            -seed ${seed} \
            --opt_args "${opt_args}" \
            --use_hyperion_dirs \
            --amp \
            
            nvflare simulator ${target_config} -w ${RESULT_ROOT}/${name} -n ${num_clients} -gpu ${gpu}
        done
    done
done