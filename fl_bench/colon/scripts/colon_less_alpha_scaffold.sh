conda activate flare
RESULT_ROOT=/path/to/sharedscratch/miccai23/colon_new
algorithms_dir="/path/to/FL_bench/fl_bench/colon/job_configs"

config=federated
gpu=0,0,0
num_clients=3
scheduler=cosine
optim_resume=resume
steps_or_epochs=steps
steps=250
rounds=20
lr=1e-5

grad_clipping_type=none
grad_clipping_value=1.0
batch_size=128
fedprox_mu=0.00

for seed in 1 2 3; do
    for method in scaffold; do
        for model_name in resnet_50_supervised nf_resnet50_supervised resnet50_gn_supervised seresnet_50_supervised nf_seresnet50_pretrained; do

            name="${model_name}_${method}_seed_${seed}"
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
            --opt_args '{"class": "AdamW", "weight_decay": 1e-3}' \
            --use_hyperion_dirs \
            --amp \
            
            nvflare simulator ${target_config} -w ${RESULT_ROOT}/${name} -n ${num_clients} -gpu ${gpu}
        done
    done
done