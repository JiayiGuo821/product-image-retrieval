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


def parse_args():
    model_names=['Matcher', 'PCA_Matcher']
    dataset_names=['Shopee_product_matching']
    text_type_names = ['tfidf', 'bert']

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='None')
    parser.add_argument('--mode', default='None')

    parser.add_argument('--model', default='Matcher', choices=model_names, help='default:Matcher')

    parser.add_argument('--batch_size','-b', type=int,metavar='N',help='default: 16')
    parser.add_argument('--dataset', default=None, choices=dataset_names)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--text_type', choices=text_type_names)
    parser.add_argument('--image_type', default='image')

    args=parser.parse_args()

    return args

def test_pca(args):
    args = vars(args)
    name = args['name']

    args['path'] = f'./results/{name}'
    path = f'./results/{name}'
    load_path = f'./results/{name}/{name}'

    print(f"=> creating model {name}")
    print('Config -----')
    for arg in args:
        print(f'{arg}: {args[arg]}')
    print('------------')

    

    file = open(os.path.join(path, 'pca.pkl'), 'rb')
    args['pca'] = pickle.load(file)

    file = open(os.path.join(path, 'pca_title.pkl'), 'rb')
    args['pca_title'] = pickle.load(file)

    
 

    args['testset'] = vars(datasets)[args['dataset']](mode=args['mode'],
            image_type=args['image_type'],
            text_type=args['text_type'],
            debug=args['debug'],
            input_size=64,
            )
    
    args['testloader'] = torch.utils.data.DataLoader(args['testset'],
            batch_size=args['batch_size'], 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            )
    
    
    model = vars(models)[args['model']](args)

    log = pd.DataFrame(index=[],
            columns=['ap',
                'roc',
                'best_component',
                'f1', 
                'threshold',
                'ap_title',
                'roc_title',
                'best_component_title',
                'f1_title',
                'threshold_title',
                'f1_merge'],
            )
    test_log = model.test(mode=args['mode'],
            image_type=args['image_type'],
            text_type=args['text_type'])

    print(f"ap: {test_log['ap']:.4f}")
    print(f"roc: {test_log['roc']:.4f}")
    print(f"best_component: {test_log['best_component']}")
    print(f"f1: {test_log['f1']:.4f}")
    print(f"threshold: {test_log['threshold']:.4f}")
    print(f"ap_title: {test_log['ap_title']:.4f}")
    print(f"roc_title: {test_log['roc_title']:.4f}")
    print(f"best_component_title: {test_log['best_component_title']}")
    print(f"f1_title: {test_log['f1_title']:.4f}")
    print(f"threshold_title: {test_log['threshold_title']:.4f}")
    print(f"f1_merge: {test_log['f1_merge']:.4f}")

    tmp = pd.Series([test_log['ap'],
        test_log['roc'],
        test_log['best_component'],
        test_log['f1'],
        test_log['threshold'],
        test_log['ap_title'],
        test_log['roc_title'],
        test_log['best_component_title'],
        test_log['f1_title'],
        test_log['threshold_title'],
        test_log['f1_merge']],
        index=['ap',
            'roc',
            'best_component',
            'f1', 
            'threshold',
            'ap_title',
            'roc_title',
            'best_component_title',
            'f1_title',
            'threshold_title',
            'f1_merge'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv(f'{load_path}_{args["mode"]}_log.csv', index=False)

if __name__ == '__main__':
    torch.cuda.empty_cache()

    args = parse_args()

    args.model = 'PCA_Matcher'
    args.dataset = 'Shopee_product_matching'
    args.batch_size = 1024
    args.debug = False



    test_pca(args)


















