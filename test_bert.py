from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import os
import torch.utils.data as data
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('-s', '--set', metavar='DIR', default='test',
                    choices=['train', 'valid', 'test'],
                    help='valid set or test set')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--name', default='', type=str,
                    help='experiment name')

class product_image_retrieval(data.Dataset):
    def __init__(self, root, mode='train', debug=False, transform=None):
        self.root = root
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
        self.corpus = list(data['title'])

    def __getitem__(self, index):
        # print(type(self.corpus[index]))
        return self.corpus[index], index

    def __len__(self):
        return len(self.corpus)


def main():
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertModel.from_pretrained('bert-large-uncased')
    model.eval().cuda(args.gpu)

    datadir = args.data

    pir_dataset = product_image_retrieval(
        datadir,
        args.set,
        )

    loader = torch.utils.data.DataLoader(
        pir_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    length_dir = len(pd.read_csv(os.path.join(datadir, 'splitted', args.set) + '.csv'))
    feats = np.zeros((length_dir, 1024))

    text = 'pattern recognition is so hard for me'
    for i, (texts, index) in tqdm(enumerate(loader)):
        # if args.gpu is not None:
        #     text = text.cuda(args.gpu, non_blocking=True)
        tokens_lst = []
        for text in texts:
            tokenized_text = tokenizer.tokenize(text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_lst.append(indexed_tokens)
        tokens_tensor = torch.tensor(tokens_lst).cuda(args.gpu, non_blocking=True)

        feats_batch = model(tokens_tensor, output_all_encoded_layers=False)[1]
        feats_batch = feats_batch.detach().cpu().numpy()

        for idx in range(len(index)):
            feats[index[idx], :] = feats_batch[idx, :]

    np.save('expr/features_bert' + args.name + '_' + args.set + '.npy', feats)


if __name__ == '__main__':
    main()