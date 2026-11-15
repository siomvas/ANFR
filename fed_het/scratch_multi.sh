 python train_FedAVG.py --FL_platform scratch-ResNet-FedAVG --dataset cifar10 --optimizer_type adamw --learning_rate 0.001 --decay_type step --warmup_steps 0 --step_size 100000 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type central --seed 98 --n 1
python train_SCAFFOLD.py --FL_platform scratch-ResNet-SCAFFOLD --dataset cifar10 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type split_2 --seed 2024 --n 1

python train_FedAVG.py --FL_platform scratch-ANFR-FedAVG --dataset cifar10 --optimizer_type adamw --learning_rate 0.001 --decay_type step --warmup_steps 0 --step_size 100000 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type central --seed 98 --n 1
python train_SCAFFOLD.py --FL_platform scratch-ANFR-SCAFFOLD --dataset cifar10 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type split_2 --seed 2024 --n 1

python train_FedAVG.py --FL_platform scratch-NF-FedAVG --dataset cifar10 --optimizer_type adamw --learning_rate 0.001 --decay_type step --warmup_steps 0 --step_size 100000 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type central --seed 98 --n 1
python train_SCAFFOLD.py --FL_platform scratch-NF-SCAFFOLD --dataset cifar10 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type split_2 --seed 2024 --n 1

python train_FedAVG.py --FL_platform scratch-ResNet-FedAVG --norm GN --dataset cifar10 --optimizer_type adamw --learning_rate 0.001 --decay_type step --warmup_steps 0 --step_size 100000 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type central --seed 98 --n 1
python train_SCAFFOLD.py --FL_platform scratch-ResNet-SCAFFOLD --norm GN --dataset cifar10 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type split_2 --seed 2024 --n 1

python train_FedAVG.py --FL_platform scratch-SE-FedAVG --dataset cifar10 --optimizer_type adamw --learning_rate 0.001 --decay_type step --warmup_steps 0 --step_size 100000 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type central --seed 98 --n 1
python train_SCAFFOLD.py --FL_platform scratch-SE-SCAFFOLD --dataset cifar10 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type split_2 --seed 2024 --n 1
