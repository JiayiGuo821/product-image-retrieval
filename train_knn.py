import torch

from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier
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
    model_names = ['KNN_Matcher']
    dataset_names=['Shopee_product_matching']
    text_type_names = ['tfidf', 'bert']

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='None')
    parser.add_argument('--model', default='Classifier', choices=model_names, help='default:Classifier')
    parser.add_argument('--dataset', default=None, choices=dataset_names)
    parser.add_argument('--debug', default=False, action='store_true')

    parser.add_argument('--text_type', choices=text_type_names)
    parser.add_argument('--image_type', default='image')

    args=parser.parse_args()

    return args

def train_knn(args):
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

    args['knn'] = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
    args['knn_title'] = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
    
    
    trainset = vars(datasets)[args['dataset']](mode='train',
            image_type=args['image_type'],
            text_type=args['text_type'],
            debug=args['debug'],
            input_size=64,
            )



    args['trainloader'] = torch.utils.data.DataLoader(trainset,
            batch_size=len(trainset),
            shuffle=True,
            num_workers=4,
            pin_memory=True)

    model = vars(models)[args['model']](args)
    model.train(args)

    #save model
    with open(os.path.join(path, 'knn.pkl'), 'wb') as f:
        pickle.dump(args['knn'], f)

    with open(os.path.join(path, 'knn_title.pkl'), 'wb') as f:
        pickle.dump(args['knn_title'], f)

if __name__ == '__main__':
    args = parse_args()

    args.dataset = 'Shopee_product_matching'
    args.model = 'KNN_Matcher'

    args.debug = False

    train_knn(args)









       





