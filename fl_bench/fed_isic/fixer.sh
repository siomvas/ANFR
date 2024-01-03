conda activate flare
cd /path/to/FL_bench/fl_bench/fed_isic
RESULT_ROOT=/path/to/localscratch/Fed_ISIC/
algorithms_dir="${PWD}/job_configs"

for method in fedopt fedavg; do
    for model_name in vit_b_16_scratch; do
        for seed in 24 42; do
            name="fixvit_fedisic_${method}_${model_name}_${seed}"
            python3 /path/to/FL_bench/fl_bench/fed_isic/src/setup.py \
                -job ${algorithms_dir}/fed_isic_federated \
                -method ${method} \
                -num_rounds 20 \
                -lr 5e-4 \
                --batch_size 64 \
                -local_steps 600 \
                -model_name ${model_name} \
                -seed ${seed} \
                --amp \
                --opt_args '{"class": "Adam"}'
            nvflare simulator ${algorithms_dir}/fed_isic_federated -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1
        done
    done
done