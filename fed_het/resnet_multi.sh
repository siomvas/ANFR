python train_FedOpt.py --FL_platform ResNet-FedOpt --dataset cifar10 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type split_2 --seed 42 --n 1

python train_SCAFFOLD.py --FL_platform ResNet-SCAFFOLD --dataset cifar10 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type split_2 --seed 42 --n 1

python train_FedAVG.py --FL_platform ResNet-FedAVG --dataset cifar10 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type split_2 --seed 2024 --n 1

python train_FedProx.py --FL_platform ResNet-FedProx --mu 0.1 --dataset cifar10 --local_epochs 1 --max_communication_rounds 100  --num_local_clients -1 --split_type split_2 --seed 2024 --n 1

python train_FedOpt.py --FL_platform ResNet-FedOpt --dataset cifar10 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type split_2 --seed 2024 --n 1

python train_SCAFFOLD.py --FL_platform ResNet-SCAFFOLD --dataset cifar10 --local_epochs 1 --max_communication_rounds 100 --num_local_clients -1 --split_type split_2 --seed 2024 --n 1