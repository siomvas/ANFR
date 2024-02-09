conda activate flare
cd /path/to/FL_bench/fl_bench/fed_isic
RESULT_ROOT=/path/to/localscratch/Fed_ISIC/new_models
algorithms_dir="${PWD}/job_configs"

seed=24
method=scaffold
init=supervised
steps=400
rounds=80

model_name=caformer
name="${init}_${method}_${model_name}_${seed}_v2"
python3 ${PWD}/src/setup.py \
-job ${algorithms_dir}/fed_isic_federated_v2 \
-method ${method} \
-num_rounds ${rounds} \
-lr 5e-4 \
--batch_size 32 \
-local_steps ${steps} \
-model_name ${model_name}_${init} \
-seed ${seed} \
--amp \
--opt_args '{"class": "Adam"}'
nvflare simulator ${algorithms_dir}/fed_isic_federated_v2 -w ${RESULT_ROOT}/${name} -n 6 -gpu 0,0,0,1,1,1


