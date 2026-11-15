data_path=./data/

for grad_clip in True; do
    for FL_platform in ViT-FedAvg; do
        for batch_size in 32; do
            python train_FedAVG.py \
            --data_path ${data_path} --dataset cifar10 --split_type split_2 --seed 98 --n 1 --valid_type orig \
            --FL_platform ${FL_platform}_${batch_size}_${grad_clip} \
            --grad_clip ${grad_clip} --batch_size ${batch_size}
        done
    done
done
