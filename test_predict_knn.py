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

    parser.add_argument('--dataset', default=None, choices=dataset_names)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--text_type', choices=text_type_names)
    parser.add_argument('--image_type', default='image')

    args=parser.parse_args()

    return args

def test_knn(args):
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

    

    file = open(os.path.join(path, 'knn.pkl'), 'rb')
    args['knn'] = pickle.load(file)

    file = open(os.path.join(path, 'knn_title.pkl'), 'rb')
    args['knn_title'] = pickle.load(file)

    
 

    args['testset'] = vars(datasets)[args['dataset']](mode=args['mode'],
            image_type=args['image_type'],
            text_type=args['text_type'],
            debug=args['debug'],
            input_size=64,
            )
    
    args['testloader'] = torch.utils.data.DataLoader(args['testset'],
            batch_size=len(args['testset']), 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            )
    
    model = vars(models)[args['model']](args)

    log = pd.DataFrame(index=[],
            columns=['f1', 
                'f1_title',
                'f1_merge'],
            )
    test_log = model.test(mode=args['mode'],
            image_type=args['image_type'],
            text_type=args['text_type'])

    print(f"f1: {test_log['f1']:.4f}")
    print(f"f1_title: {test_log['f1_title']:.4f}")
    print(f"f1_merge: {test_log['f1_merge']:.4f}")

    tmp = pd.Series([test_log['f1'],
        test_log['f1_title'],
        test_log['f1_merge']],
        index=['f1', 
            'f1_title',
            'f1_merge'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv(f'{load_path}_{args["mode"]}_log.csv', index=False)

if __name__ == '__main__':
    torch.cuda.empty_cache()

    args = parse_args()

    args.model = 'KNN_Matcher'
    args.dataset = 'Shopee_product_matching'
    args.debug = False



    test_knn(args)


















