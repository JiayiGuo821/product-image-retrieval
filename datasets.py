import torch.utils.data as data
import pandas as pd
import numpy as np
import os

from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from torchvision import transforms
from PIL import Image

class Shopee_product_matching(data.Dataset):
    def __init__(self, mode: str, image_type, text_type, transform=None, debug=False, input_size=224):
        self.root = './data'
        self.expr_root = './expr'
        self.images = []
        self.labels = []
        self.mode = mode
        self.text_type = text_type
        self.image_type = image_type

        if image_type == 'image':
            mean_pix = [0.485, 0.456, 0.406]
            std_pix = [0.229, 0.224, 0.225]

            if transform == None:
                self.transform = transforms.Compose([
                    transforms.Resize((input_size,input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean_pix, std=std_pix),
                    ])
            else:
                self.transform = transform


            csv_file = os.path.join(self.root, 'splitted', f'{mode}.csv') #test set
            data = pd.read_csv(csv_file)
            if debug:
                data = data.iloc[:100]
            data['image'] = data['image'].apply(lambda image:os.path.join(self.root, 'train_images', image))
            self.images = list(data['image'])
        else:
            try:
                self.images = np.load(os.path.join(self.expr_root, f'{image_type}.npy'))
            except:
                print('NO Features!')

        try:
            self.labels = np.load(os.path.join(self.expr_root, f'labels_{mode}.npy'))
        except:
            vc = list(set(pd.read_csv(os.path.join(self.root, 'train.csv'))['label_group']))
            vc.sort()
            group2label = dict(zip(vc,range(len(vc))))
            import operator
            data = pd.read_csv(os.path.join(self.root, 'splitted', f'{mode}.csv'))
            self.labels = operator.itemgetter(*list(data['label_group']))(group2label)
            np.save(os.path.join(self.expr_root, f'labels_{mode}.npy'), self.labels)

        if self.text_type == 'bert':
            self.features_title = np.load(os.path.join(self.expr_root, f'features_bert_{mode}.npy'))
        elif self.text_type == 'tfidf':
            csv_file = os.path.join(self.root, 'splitted', f'{mode}.csv') #test set
            data = pd.read_csv(csv_file)
            if debug:
                data = data.iloc[:100]
            corpus = list(data['title'])
            vector = TfidfVectorizer()
            tfidf = vector.fit_transform(corpus)
            self.features_title = tfidf.toarray()        
        elif self.text_type == 'bm25':
            csv_file = os.path.join(self.root, 'splitted', f'{mode}.csv') #test set
            data = pd.read_csv(csv_file)
            if debug:
                data = data.iloc[:100]
            corpus = list(data['title'])
            tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            tokenized_corpus = [tokenizer.tokenize(doc) for doc in corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.features_title = tokenized_corpus

    def __getitem__(self, index):
        if self.image_type == 'image':
            image_path = self.images[index]
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
        else:
            img = self.images[index]

        target = int(self.labels[index])

        if self.text_type == 'bm25':
            tokenized_query = self.features_title[index]
            feature_title = self.bm25.get_scores(tokenized_query)
        else:
            feature_title = self.features_title[index]
        return img, target, feature_title
    def __len__(self):
        return len(self.labels)
if __name__ == '__main__':
    dataset = Shopee_product_matching(mode='test', text_type='bm25', debug=True)
    #trainset = FashionMNIST(train=True)
    #for img, target in dataset:
    #    print(img, target)
    #    break
   
    #trainset = MiniImageNetL(r=72, r1=0, vis=True, low=True, debug=False, train=False)
    #low_path = './results/detail/low_detail_results(r=72).csv'
    #low = pd.read_csv(low_path).loc[:,'acc']
    #low = np.array(low)
    #print (np.sum(low))
    #low = ~low.astype(np.bool)
    #
    #std_path = './results/detail/std_detail_results.csv'
    #std = pd.read_csv(std_path).loc[:,'acc']
    #std = np.array(std)
    #print (np.sum(std))
    #std = std.astype(np.bool)
    #
    #for i in np.argwhere((std*low)==True).squeeze():
    #    trainset[i]



    #for i in range(1):
        #index = np.random.permutation(np.array(range(10000)))[i]
        #trainset[index]

    




