#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data as data

import moco.builder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('-s', '--set', metavar='DIR', default='test',
                    choices=['valid', 'test'],
                    help='valid set or test set')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')


# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')

class product_image_retrieval(data.Dataset):
    def __init__(self, root, mode='train', debug=False, transform=None):
        self.root = root
        self.images = []
        self.labels = []
        self.transform = transform

        if mode == 'train':
            csv_file = os.path.join(self.root, 'splitted', 'train.csv')
        elif mode == 'valid':
            csv_file = os.path.join(self.root, 'splitted', 'valid.csv')
        elif mode == 'test':
            csv_file = os.path.join(self.root, 'splitted', 'test.csv')
        else:
            print('mode must in train, valid or test')
            raise NotImplementedError

        data = pd.read_csv(csv_file)
        if debug:
            data = data.iloc[:20]
        corpus = list(data['title'])
        vector = TfidfVectorizer()
        print('preparing tfidf...')
        tfidf = vector.fit_transform(corpus)
        self.weightlist = tfidf.toarray()
        np.save(os.path.join(self.root, 'splitted', mode + '_tfidf.npy'), self.weightlist)
        print('tfidf ready...')
        data['image'] = data['image'].apply(lambda image: os.path.join(self.root, 'train_images', image))
        vc = list(set(data['label_group']))
        vc.sort()
        group2label = dict(zip(vc, range(len(vc))))
        import operator
        self.images = list(data['image'])
        self.labels = operator.itemgetter(*list(data['label_group']))(group2label)
        data['label_group'] = self.labels
        self.data = data

    def __getitem__(self, index):
        image_path = self.images[index]
        img = Image.open(image_path).convert('RGB')
        target = int(self.labels[index])
        title = self.weightlist[index]

        if self.transform is not None:
            img = self.transform(img)

        sample = {'img': img, 'title': title, 'target': target, 'index': index}  # 根据图片和标签创建字典

        return sample

    def __len__(self):
        return len(self.labels)

def main():
    args = parser.parse_args()

    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo_single(
        models.__dict__[args.arch],
        args.moco_dim, args.mlp)
    print(model)
    for name, param in model.named_parameters():
        param.requires_grad = False

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            model.load_state_dict(state_dict, strict=True)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    datadir = args.data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    pir_dataset = product_image_retrieval(
        datadir,
        args.set,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    loader = torch.utils.data.DataLoader(
        pir_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    length_dir = len(pd.read_csv(os.path.join(datadir, 'splitted', args.set) + '.csv'))
    feats = np.zeros((length_dir, args.moco_dim))

    model.eval()
    for i, samples in tqdm(enumerate(loader)):
        if args.gpu is not None:
            img = samples['img'].cuda(args.gpu, non_blocking=True)
            title = samples['title'].cuda(args.gpu, non_blocking=True)
            target = samples['target'].cuda(args.gpu, non_blocking=True)
            index = samples['index'].cuda(args.gpu, non_blocking=True)

        feats_batch = model(im_q=img, title=title, target=target).cpu().numpy()

        for idx in range(len(index)):
            feats[index[idx], :] = feats_batch[idx, :]

    np.save('expr/feats_' + args.set + '.npy', feats)

if __name__ == '__main__':
    main()
