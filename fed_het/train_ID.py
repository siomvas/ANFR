# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils.data_utils import DatasetFLViT, create_dataset_and_evalmetrix
from utils.util import Partial_Client_Selection, valid, valid_alt, average_model, optimizer_to
from utils.start_config import initization_configure
from typing import List, Tuple, Union, OrderedDict
import skdim 
import pandas as pd
import ast
from torch.utils.tensorboard import SummaryWriter
from spectral_metric.estimator import CumulativeGradientEstimator

def estimate_intrinsic_dimension(model, data_loader, device, logger, current_round, name="Global", client_id=None):
        """Estimate the intrinsic dimension of the model on the local data."""
        features = []
        normalized_features = []
        label_list = []
        with torch.no_grad():
            # make copy of model and reset it's classifier head to make it a feature extractor
            model_eval = deepcopy(model).eval()
            model_eval.reset_classifier(0)
            for batch in data_loader:
                images = batch[0].to(device)
                labels = batch[1]
                output = model_eval(images)
                normal_output = torch.nn.functional.normalize(output, dim=1)
                features.append(output.cpu().numpy())
                normalized_features.append(normal_output.cpu().numpy())
                label_list.append(labels.numpy())
            features = np.concatenate(features, axis=0)
            normalized_features = np.concatenate(normalized_features, axis=0)
            label_list = np.concatenate(label_list, axis=0)
            estimator = CumulativeGradientEstimator(M_sample=50, k_nearest=3)
            estimator.fit(normalized_features, label_list)
            csg = estimator.csg.item() if not np.isnan(estimator.csg) else np.nan
            logger.add_scalar(
                f"Intrinsic Dimension {name}/CSG",
                csg,
                current_round,
            )
            print(f"CSG dimension: {csg}")
            twonn_estimator = skdim.id.TwoNN().fit(features, label_list)
            twonn_dimension = twonn_estimator.dimension_
            logger.add_scalar(
                f"Intrinsic Dimension {name}/TwoNN",
                twonn_dimension,
                current_round,
            )

            print(f"TwoNN dimension: {twonn_dimension}")

            mle_estimator = skdim.id.MLE().fit(features, label_list)
            mle_dimension = mle_estimator.dimension_

            logger.add_scalar(
                f"Intrinsic Dimension {name}/MLE",
                mle_dimension,
                current_round,
            )

            print(f"MLE dimension: {mle_dimension}")

            # FisherS estimator

            fisher_estimator = skdim.id.FisherS().fit(features, label_list)
            fisher_dimension = fisher_estimator.dimension_
 
            print(f"FisherS dimension: {fisher_dimension}")
            logger.add_scalar(
                f"Intrinsic Dimension {name}/FisherS",
                fisher_dimension,
                current_round,
            )
            # clear the features and labels
            features = None
            label_list = None
            normalized_features = None
            model_eval = None
        
        return csg, twonn_dimension, mle_dimension, fisher_dimension

def train(args, model): 
    """ Train the model """

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare dataset
    loaded_npy = create_dataset_and_evalmetrix(args)

    # prepare record for global model ID metrics. One row per round 4 columns per client: CSG, TwoNN, MLE, FisherS
    global_id_metrics = pd.DataFrame(
        columns=[
            f"{client}_CSG" for client in args.dis_cvs_files
        ] + [
            f"{client}_TwoNN" for client in args.dis_cvs_files
        ] + [
            f"{client}_MLE" for client in args.dis_cvs_files
        ] + [
            f"{client}_FisherS" for client in args.dis_cvs_files
        ]
    )
    id_metrics_curr_global = {}

    # prepare record for local model ID metrics. One row per round 4 columns per client: CSG, TwoNN, MLE, FisherS
    local_id_metrics = pd.DataFrame(
        columns=[
            f"{client}_CSG" for client in args.dis_cvs_files
        ] + [
            f"{client}_TwoNN" for client in args.dis_cvs_files
        ] + [
            f"{client}_MLE" for client in args.dis_cvs_files
        ] + [
            f"{client}_FisherS" for client in args.dis_cvs_files
        ]
    )
    id_metrics_curr_local = {}

    print('Loading testset, phase test')
    testset = DatasetFLViT(args, loaded_npy, phase = 'test')
    test_loader = DataLoader(testset, sampler=SequentialSampler(testset), batch_size=args.batch_size, num_workers=args.num_workers)


    # if not celeba then get the union val dataset,
    if args.dataset not in ['celeba', 'gldk23', 'isic19']:
        print('Loading valset, phase val')
        valset = DatasetFLViT(args, loaded_npy, phase = 'val')
        val_loader = DataLoader(valset, sampler=SequentialSampler(valset), batch_size=args.batch_size, num_workers=args.num_workers)

    # Configuration for FedAVG, prepare model, optimizer, scheduler
    model_all, optimizer_all, scheduler_all = Partial_Client_Selection(args, model)
    model_avg = deepcopy(model).cpu()

    logger = SummaryWriter(os.path.join(args.output_dir, 'tb_logs'), flush_secs=5)

    # Train
    print("=============== Running training ===============")
    loss_fct = torch.nn.CrossEntropyLoss()
    tot_clients = args.dis_cvs_files
    epoch = -1


    while True:
        epoch += 1
        # randomly select partial clients
        if args.num_local_clients == len(args.dis_cvs_files):
            # just use all the local clients
            cur_selected_clients = args.proxy_clients
        else:
            cur_selected_clients = np.random.choice(tot_clients, args.num_local_clients, replace=False).tolist()

        # Get the quantity of clients joined in the FL train for updating the clients weights
        cur_tot_client_Lens = 0
        for client in cur_selected_clients:
            cur_tot_client_Lens += args.clients_with_len[client]

        val_loader_proxy_clients = {}

        for cur_single_client, proxy_single_client in zip(cur_selected_clients, args.proxy_clients):
            args.single_client = cur_single_client
            args.clients_weightes[proxy_single_client] = args.clients_with_len[cur_single_client] / cur_tot_client_Lens

            print('Loading trainset, phase train')
            trainset = DatasetFLViT(args, loaded_npy, phase='train')
            train_loader = DataLoader(trainset, sampler=RandomSampler(trainset), batch_size=args.batch_size, num_workers=args.num_workers)

            if args.dataset == 'celeba' or  args.dataset == 'gldk23' or args.dataset == 'isic19':
                valset = DatasetFLViT(args, loaded_npy, phase='val')
                val_loader_proxy_clients[proxy_single_client] = DataLoader(valset, sampler=SequentialSampler(valset), batch_size=args.batch_size,
                                          num_workers=args.num_workers)
            else:
                # for Cifar10 datasets we use union validation dataset
                val_loader_proxy_clients[proxy_single_client] = val_loader

            model = model_all[proxy_single_client]
            optimizer = optimizer_all[proxy_single_client]
            scheduler = scheduler_all[proxy_single_client]

            model = model.to(args.device).train()
            optimizer_to(optimizer, args.device)

            if args.decay_type == 'step':
                scheduler.step()
            
            print("Estimating global model's intrinsic dimension for client {}".format(cur_single_client))
            id_metrics_curr_global[cur_single_client] = estimate_intrinsic_dimension(model, train_loader, args.device, logger, epoch, name="Global", client_id=cur_single_client)
            print('Train the client', cur_single_client, 'of communication round', epoch)

        

            for inner_epoch in range(args.local_epochs):
                for step, batch in enumerate(train_loader):  
                    args.global_step_per_client[proxy_single_client] += 1
                    batch = tuple(t.to(args.device) for t in batch)

                    x, y = batch
                    predict = model(x)
                    loss = loss_fct(predict.view(-1, args.num_classes), y.view(-1))

                    loss.backward()

                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if not args.decay_type == 'step':
                        scheduler.step()
                    args.learning_rate_record[proxy_single_client].append(optimizer.param_groups[0]['lr'])

                    if (step+1 ) % 10 == 0:
                        print(cur_single_client, step,':', len(train_loader),'inner epoch', inner_epoch, 'round', epoch,':',
                              args.max_communication_rounds, 'loss', loss.item(), 'lr', optimizer.param_groups[0]['lr'])
            
            print("Estimating local model's intrinsic dimension for client {}".format(cur_single_client))
            id_metrics_curr_local[cur_single_client] = estimate_intrinsic_dimension(model, train_loader, args.device, logger, epoch, name="Local", client_id=cur_single_client)


            # we use frequent transfer of model between GPU and CPU due to limitation of GPU memory
            model.to('cpu')
            optimizer_to(optimizer, 'cpu')

        # average_model(args, model_avg, model_all) # updates client model param

        # then evaluate
        for cur_single_client, proxy_single_client in zip(cur_selected_clients, args.proxy_clients):
            args.single_client = cur_single_client
            model = model_all[proxy_single_client]
            model.to(args.device)
            if args.valid_type == 'orig':
                valid(args, model, val_loader_proxy_clients[proxy_single_client], test_loader, TestFlag=True)
            elif args.valid_type == 'alt':
                valid_alt(args, model, val_loader_proxy_clients[proxy_single_client], test_loader, TestFlag=True)
            model.cpu()

        args.record_val_acc = pd.concat([args.record_val_acc, pd.DataFrame([args.current_acc])], ignore_index=True)
        args.record_val_acc.to_csv(os.path.join(args.output_dir, 'val_acc.csv'))

        args.record_test_acc  = pd.concat([args.record_test_acc, pd.DataFrame([args.current_test_acc])], ignore_index=True)
        args.record_test_acc.to_csv(os.path.join(args.output_dir, 'test_acc.csv'))

        np.save(args.output_dir + '/learning_rate.npy', args.learning_rate_record)

        # save test acc
        tmp_round_acc = [val for val in args.current_test_acc.values() if type(val) != list]
        scalar_test_acc = np.asarray(tmp_round_acc).mean()
        # save al acc 
        tmp_round_val_acc =  [val for val in args.current_acc.values() if type(val) != list]
        scalar_val_acc = np.asarray(tmp_round_val_acc).mean()

        # save the global model ID metrics
        global_id_metrics = pd.concat([global_id_metrics, pd.DataFrame([id_metrics_curr_global[cur_single_client] for cur_single_client in cur_selected_clients], index=cur_selected_clients)], ignore_index=True)
        global_id_metrics.to_csv(os.path.join(args.output_dir, 'global_id_metrics.csv'))
        id_metrics_curr_global = {}
        # save the local model ID metrics
        local_id_metrics = pd.concat([local_id_metrics, pd.DataFrame([id_metrics_curr_local[cur_single_client] for cur_single_client in cur_selected_clients], index=cur_selected_clients)], ignore_index=True)
        local_id_metrics.to_csv(os.path.join(args.output_dir, 'local_id_metrics.csv'))
        id_metrics_curr_local = {}
        
        print("Epoch {}: Avg test acc {}, Avg Val acc {}".format(epoch, scalar_test_acc, scalar_val_acc))

        if args.global_step_per_client[proxy_single_client] >= args.t_total[proxy_single_client]:
            break



    print("================End training! ================ ")


def main():
    parser = argparse.ArgumentParser()
    # General DL parameters
    parser.add_argument("--FL_platform", type = str, default="ResNet-FedAVG",  help="Choose of different FL platform. ")
    parser.add_argument("--norm", type = str, default=None,  help="Selects a normalization layer for the model. Options: BN, LN, GN")
    parser.add_argument("--valid_type", choices=["orig", "alt"], default="orig", help="Selects whether test accuracy is recorded when the validation accuracy has increased (orig) or always (alt).")    
    parser.add_argument("--dataset", choices=["cifar10", "celeba", "pacs", "gldk23", "isic19", "domainnet"], default="cifar10", help="Which dataset.")
    parser.add_argument("--data_path", type=str, default='./data/', help="Where is dataset located.")

    parser.add_argument("--save_model_flag",  action='store_true', default=False,  help="Save the best model for each client.")

    parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints/results/logs will be written.")
    parser.add_argument("--optimizer_type", default="adam",choices=["sgd","adam", "adamw"], type=str, help="Ways for optimization.")
    parser.add_argument("--num_workers", default=8, type=int, help="num_workers")
    parser.add_argument("--weight_decay", default=0, choices=[0.05, 0], type=float, help="Weight deay if we apply some. 0 for SGD and 0.05 for AdamW in paper")
    parser.add_argument('--grad_clip', type=ast.literal_eval, default=True, help="whether gradient clip to 1 or not")

    parser.add_argument("--img_size", default=224, type=int, help="Final train resolution")
    parser.add_argument("--batch_size", default=32, type=int,  help="Local batch size for training.")
    parser.add_argument("--gpu_ids", type=str, default='0', help="gpu ids: e.g. 0  0,1,2")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--n', type=int, default=0, help = "nth repetition of the same experiment")

    ## section 2:  DL learning rate related
    parser.add_argument("--decay_type", choices=["cosine", "linear", "step"], default="cosine",  help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Step of training to perform learning rate warmup for if set for cosine and linear deacy.")
    parser.add_argument("--step_size", default=30, type=int, help="Period of learning rate decay for step size learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,  help="Max gradient norm.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,  help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")

    ## FL related parameters
    parser.add_argument("--local_epochs", default=5, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=20, type=int,  help="Total communication rounds")
    parser.add_argument("--num_local_clients", default=-1, choices=[10, 20, -1], type=int, help="Num of local clients joined in each FL train. -1 indicates all clients")
    parser.add_argument("--split_type", type=str, choices=["split_1", "split_2", "split_3", "real", "central"], default="split_3", help="Which data partitions to use")


    args = parser.parse_args()

    # Initialization

    model = initization_configure(args)

    # Training, Validating, and Testing
    train(args, model)


    message = '\n \n ==============Start showing final performance ================= \n'
    message += 'Final union test accuracy is: %2.5f  \n' %  \
                   (np.asarray(list(args.current_test_acc.values())).mean())
    message += "================ End ================ \n"


    with open(args.file_name, 'a+') as args_file:
        args_file.write(message)
        args_file.write('\n')

    print(message)


if __name__ == "__main__":
    main()
