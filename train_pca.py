import torch

from sklearn.decomposition import IncrementalPCA
from torchvision import transforms
import pandas as pd
import numpy as np
import argparse
import pickle
import sys
import os 

import models
import datasets

def parse_args():
    model_names = ['PCA_Matcher']
    dataset_names=['Shopee_product_matching']
    text_type_names = ['tfidf', 'bert']

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='None')
    parser.add_argument('--model', default='Classifier', choices=model_names, help='default:Classifier')
    parser.add_argument('--batch_size','-b', default=16, type=int, metavar='N',help='default: 16')
    parser.add_argument('--dataset', default=None, choices=dataset_names)

    parser.add_argument('--debug', default=False, action='store_true')

    parser.add_argument('--n_components','-n', type=int)
    parser.add_argument('--n_components_title','-n_t', type=int)

    parser.add_argument('--text_type', choices=text_type_names)
    parser.add_argument('--image_type', default='image')

    args=parser.parse_args()

    return args

def train_pca(args):
    args = vars(args)
    name = args['name']
    args['path'] = f'./results/{name}'
    path = args['path']
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

    args['pca'] = IncrementalPCA(n_components=args['n_components'], batch_size=args['batch_size'])
    args['pca_title'] = IncrementalPCA(n_components=args['n_components_title'], batch_size=args['batch_size'])
    
    
    trainset = vars(datasets)[args['dataset']](mode='train',
            image_type=args['image_type'],
            text_type=args['text_type'],
            debug=args['debug'],
            input_size=64,
            )

    args['trainloader'] = torch.utils.data.DataLoader(trainset,
            batch_size=args['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True)

    model = vars(models)[args['model']](args)
    model.train(image_type=args['image_type'], text_type=args['text_type'])

    #save model
    with open(os.path.join(path, 'pca.pkl'), 'wb') as f:
        pickle.dump(args['pca'], f)

    with open(os.path.join(path, 'pca_title.pkl'), 'wb') as f:
        pickle.dump(args['pca_title'], f)

if __name__ == '__main__':
    args = parse_args()
    #args.name = 'pca_test'

    args.dataset = 'Shopee_product_matching'
    args.model = 'PCA_Matcher'

    #args.batch_size = 256

    args.debug = False

    #args.n_components = 64
    #args.n_components_title = 128

    train_pca(args)










       





