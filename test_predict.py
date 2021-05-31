from torchvision import transforms, utils
import torch.optim as optim
import torch

import pandas as pd
import numpy as np
import argparse
import pickle
import sys
import os 

import models
import datasets
import nets


def parse_args():
    net_names = ['resnet50']
    model_names = ['Matcher', 'PCA_Matcher']
    dataset_names = ['Shopee_product_matching']
    text_type_names = ['tfidf', 'bm25', 'bert']

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='None')
    parser.add_argument('--net', choices=net_names)
    parser.add_argument('--model', default='Matcher', choices=model_names, help='default:Matcher')

    parser.add_argument('--batch_size','-b',default=16, type=int,metavar='N',help='default: 16')
    parser.add_argument('--dataset', default=None, choices=dataset_names)
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--debug',  default=False, action='store_true')

    parser.add_argument('--mode', default='None')
    parser.add_argument('--text_type', choices=text_type_names)
    parser.add_argument('--image_type', default='image')

    args=parser.parse_args()

    return args


def test(args):
    args = vars(args)
    name = args['name']

    args['path'] = f'./results/{name}'
    path = args['path'] 
    load_path = f'./results/{name}/{name}'

    if not os.path.exists(path):
        os.makedirs(path)

    print(f"=> creating model {name}")
    print('Config -----')
    for arg in args:
        print(f'{arg}: {args[arg]}')
    print('------------')

    with open(os.path.join(path, 'args.txt'), 'w') as f:
        for arg in args:
            print(f'{arg}: {args[arg]}', file=f)

    args['net'] = vars(nets)[args['net']](pretrained=args['pretrained'])

    args['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    args['testset'] = vars(datasets)[args['dataset']](mode=args['mode'],
            image_type=args['image_type'],
            text_type=args['text_type'], 
            debug=args['debug'])
    args['testloader'] = torch.utils.data.DataLoader(args['testset'], 
            batch_size=args['batch_size'], 
            shuffle=False,
            pin_memory=True,
            num_workers=4)
    
    model = vars(models)[args['model']](args)

    log = pd.DataFrame(index=[],
            columns=['ap', 'roc', 'f1', 'threshold', 'ap_title', 'roc_title', 'f1_title', 'threshold_title','f1_merge'],
            )
    test_log = model.test(path=path,
            debug=args['debug'],
            image_type=args['image_type'],
            text_type=args['text_type'])

    print(f"ap: {test_log['ap']:.4f}")
    print(f"roc: {test_log['roc']:.4f}")
    print(f"f1: {test_log['f1']:.4f}")
    print(f"threshold: {test_log['threshold']:.4f}")
    print(f"ap_title: {test_log['ap_title']:.4f}")
    print(f"roc_title: {test_log['roc_title']:.4f}")
    print(f"f1_title: {test_log['f1_title']:.4f}")
    print(f"threshold_title: {test_log['threshold_title']:.4f}")
    print(f"f1_merge: {test_log['f1_merge']:.4f}")

    tmp = pd.Series([test_log['ap'],
        test_log['roc'],
        test_log['f1'],
        test_log['threshold'],
        test_log['ap_title'],
        test_log['roc_title'],
        test_log['f1_title'],
        test_log['threshold_title'],
        test_log['f1_merge']],
        index=['ap',
            'roc',
            'f1',
            'threshold',
            'ap_title',
            'roc_title', 
            'f1_title', 
            'threshold_title',
            'f1_merge'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv(f'{load_path}_test_log.csv', index=False)

    #torch.cuda.empty_cache()
    #return log


if __name__ == '__main__':
    torch.cuda.empty_cache()

    args = parse_args()

    #args.name = 'moco_test'
    #args.mode = 'test'

    args.net = 'resnet50'
    args.model = 'Matcher'
    args.dataset = 'Shopee_product_matching'
    args.batch_size = 128

    #args.debug = False
    
    args.pretrained = True


    if args.image_type == 'image':
        print("invalid choice")
        raise Exception
    else:
        test(args)


















