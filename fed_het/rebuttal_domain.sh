data_path=./data/
for optimizer_type in sgd; do
    for learning_rate in 0.03; do
        for decay_type in cosine; do
            for warmup_steps in 100; do
                for step_size in 30; do
                    for grad_clip in False; do
                        for wd in 0; do
                            for FL_platform in ResNet-GN-FedAvg-Domain; do
                                for batch_size in 64; do
                                    python train_FedAVG.py \
                                    --data_path ${data_path} --dataset domainnet --split_type real --seed 98 --n 1 --valid_type alt \
                                    --optimizer_type ${optimizer_type} --learning_rate ${learning_rate} --decay_type ${decay_type} --warmup_steps ${warmup_steps} --step_size ${step_size} \
                                    --FL_platform ${FL_platform}_${optimizer_type}_${batch_size}_${learning_rate}_${decay_type}_${warmup_steps}_${step_size}_${grad_clip}_${wd} \
                                    --grad_clip ${grad_clip} --weight_decay ${wd} --batch_size ${batch_size} --norm GN
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done