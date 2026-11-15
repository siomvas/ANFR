import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import timm

def print_options(args, model):
    message = ''

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = num_params / 1000000

    message += "================ FL train of %s with total model parameters: %2.1fM  ================\n" % (args.FL_platform, num_params)

    message += '++++++++++++++++ Other Train related parameters ++++++++++++++++ \n'

    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '++++++++++++++++  End of show parameters ++++++++++++++++ '


    ## save to disk of current log
    args.file_name = os.path.join(args.output_dir, 'log_file.txt')

    with open(args.file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')

    print(message)


def initization_configure(args, vis= False):

    args.device = torch.device("cuda:{gpu_id}".format(gpu_id = args.gpu_ids) if torch.cuda.is_available() else "cpu")

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not args.device == 'cpu':
        torch.cuda.manual_seed(args.seed)

    # select class number 
    dataset_class_map = {
        'cifar10': 10,
        'pacs': 7,
        'gldk23': 203,
        'isic19': 8,
        'celeba': 2
    }
    args.num_classes = dataset_class_map[args.dataset]


    # select and initialize model 
    if "ResNet" in args.FL_platform:

        if args.norm and 'GN' in args.norm: 

            if "scratch" not in args.FL_platform:
                print('Architecture: ResNet-50 with GN')
                model = timm.create_model('resnet50_gn.a1h_in1k', pretrained=True, num_classes=args.num_classes)
            else:
                print('Architecture: Untrained ResNet-50 with GN')
                model = timm.create_model('resnet50_gn.a1h_in1k', pretrained=False, num_classes=args.num_classes)                

        else:
            
            if "scratch" not in args.FL_platform:
                print('Architecture: ResNet-50')
                model = timm.create_model('resnet50.ra_in1k',pretrained=True, num_classes=args.num_classes)
            else:
                print('Architecture: Untrained ResNet-50')
                model = timm.create_model('resnet50', pretrained=False, num_classes=args.num_classes)

        # model.fc = nn.Linear(model.fc.weight.shape[1], args.num_classes)
        model.to(args.device)
    elif "ECA" in args.FL_platform:
        print("Architecture: ANFR with ECA")
        model = timm.create_model(
            'nf_ecaresnet50', pretrained=False,
            num_classes=args.num_classes)
        model.to(args.device)
        
    elif "CBAM" in args.FL_platform:
        print("Architecture: ANFR with CBAM")
        model = timm.create_model(
            'nf_cbamresnet50', pretrained=False,
            num_classes=args.num_classes)
        model.to(args.device)

    elif "SE" in args.FL_platform:
        if 'scratch' in args.FL_platform:
            print('Architecture: Untrained SE-ResNet-50')
            model = timm.create_model('seresnet50', pretrained=False, num_classes=args.num_classes)
            model.to(args.device)
        else:
            print('Architecture: SE-ResNet-50')
            model = timm.create_model('seresnet50.ra2_in1k', pretrained=True, num_classes=args.num_classes)
            model.to(args.device)

    elif "NF" in args.FL_platform:
        if 'scratch' in args.FL_platform:
            print('Architecture: Untrained NF-ResNet-50')
            model = timm.create_model('nf_resnet50', pretrained=False, num_classes=args.num_classes)
            model.to(args.device)
        else:
            print('Architecture: NF-ResNet-50')
            model = timm.create_model('nf_resnet50.ra2_in1k', pretrained=True, num_classes=args.num_classes)
            model.to(args.device)

    elif "ANFR" in args.FL_platform:
        if 'scratch' in args.FL_platform:
            print('Architecture: Untrained ANFR with SE')
            model = timm.create_model('nf_seresnet50', pretrained=False, num_classes=args.num_classes)
            model.to(args.device)
        else:
            print('Architecture: ANFR-ResNet-50')
            model = timm.create_model(
                'nf_seresnet50', pretrained=False,
                num_classes=args.num_classes)
            model.to(args.device)

        
    name_parts = [
        args.FL_platform,
        args.dataset,
        args.split_type,
        f"run_{args.n}",
        f"lr_{args.learning_rate}",
        f"Seed_{args.seed}"
    ]

    if args.norm:
        name_parts.insert(1, args.norm)

    args.name_run = '_'.join(name_parts)


    args.output_dir = os.path.join('output', args.FL_platform, args.dataset, args.name_run)
    os.makedirs(args.output_dir, exist_ok=True)
    print_options(args, model)

    # set train val related paramteres
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}
    args.current_test_acc_avg = {}

    return model
