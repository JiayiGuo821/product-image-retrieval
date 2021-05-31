import os
import pickle

import torch
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from scipy import sparse
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import IncrementalPCA, SparsePCA, MiniBatchSparsePCA
from tqdm import tqdm


class Matcher():
    def __init__(self, args):
        keys = ['net', 'trainloader', 'testloader', 'testset', 'batch_size', 'device']
        self.net, self.trainloader, self.testloader, self.testset, self.batch_size, self.device = [args.get(key) for key in keys]
    def test(self, path, debug, image_type, text_type):
        if image_type == 'images':
            print('no_features')
            raise Exception
        else:
            features = self.testset.images
            features_title = self.testset.features_title
            labels = self.testset.labels

        print('get dist')
        dist = pdist(features,metric='cosine')
        dist = squareform(dist)
        
        print('get dist_title')
        if text_type == 'bert':
            dist_title = pdist(features_title,metric='cosine')
            dist_title = squareform(dist_title)
        elif text_type == 'tfidf':
            A_sparse = sparse.csr_matrix(features_title)
            dist_title = 1 - cosine_similarity(A_sparse)
    
        print('get ground_truth')
        ground_truth = ((labels[:,None]-labels[None,:]) == 0)+0

        m = len(labels)
        threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98]
        threshold_title_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98]
        
        ap = []
        roc = []
        ap_title = []
        roc_title = []
        
        f1 = np.zeros((len(threshold_list),m))
        f1_title = np.zeros((len(threshold_title_list),m))
        for i in tqdm(range(m)):
            score = 1 - dist[:,i:i+1]
            ap.append(average_precision_score(ground_truth[:,i], score))
            roc.append(roc_auc_score(ground_truth[:,i], score))
            if text_type == 'bm25':
                score_title = self.testset[i][2]
            else:
                score_title = 1 - dist_title[:,i:i+1]
            ap_title.append(average_precision_score(ground_truth[:,i], score_title))
            roc_title.append(roc_auc_score(ground_truth[:,i], score_title))

        for j, threshold in enumerate(threshold_list):
            for i in tqdm(range(m)):
                score = 1 - dist[:,i:i+1]
                pred = (score > threshold) 
                f1[j,i] = f1_score(ground_truth[:,i],pred)
        f1 = np.mean(f1, axis=1)
        print(f'f1: {f1}')
                
        for j, threshold_title in enumerate(threshold_title_list):
            for i in tqdm(range(m)):
                if text_type == 'bm25':
                    score_title = self.testset[i][2]
                else:
                    score_title = 1 - dist_title[:,i:i+1]
                pred_title = (score_title > threshold_title) 
                f1_title[j,i] = f1_score(ground_truth[:,i],pred_title)
        
        f1_title = np.mean(f1_title, axis=1)
        print(f'f1_title: {f1_title}')
        
        best_threshold = threshold_list[np.argmax(f1)]
        best_threshold_title = threshold_title_list[np.argmax(f1_title)]
        
        f1_merge = []
        for i in tqdm(range(m)):
            score = 1 - dist[:,i:i+1]
            pred = (score > best_threshold) 
            if text_type == 'bm25':
                score_title = self.testset[i][2]
            else:
                score_title = 1 - dist_title[:,i:i+1]
            pred_title = (score_title > best_threshold_title) 
        
            pred_merge = pred + pred_title
            f1_merge.append(f1_score(ground_truth[:,i],pred_merge))
        
        
        test_log = OrderedDict([('ap', np.mean(ap)), 
            ('roc', np.mean(roc)), 
            ('f1', np.max(f1)),
            ('threshold', best_threshold),
            ('ap_title', np.mean(ap_title)),
            ('roc_title', np.mean(roc_title)),
            ('f1_title', np.max(f1_title)),
            ('threshold_title', best_threshold_title),
            ('f1_merge', np.mean(f1_merge))])
        return test_log

    def get_feature():
        device = self.device
        net = self.net.to(device)
        net.eval()
        testloader = self.testloader

        weightlist = []
        features = []
        labels = []
        def hook(module, input, output):
            features.append(output.cpu().numpy())
        net.avgpool.register_forward_hook(hook)
        with torch.no_grad():
            for i, (inputs,targets,weights)  in tqdm(enumerate(testloader),total=len(testloader)):
                inputs = inputs.to(device, non_blocking=True)
                net(inputs)
                labels.append(targets.numpy())
                weightlist.append(weights.numpy())


        torch.cuda.empty_cache()
        features = np.squeeze(np.row_stack(features))
        weightlist = np.squeeze(np.row_stack(weightlist))
        labels = np.squeeze(np.concatenate(labels))

        np.save(os.path.join(path,'features.npy'), features)
        np.save(os.path.join(path,'weightlist.npy'), weightlist)
        np.save(os.path.join(path,'labels.npy'), labels)

class PCA_Matcher():
    def __init__(self, args):
        keys = ['pca', 'pca_title', 'trainloader', 'testloader', 'testset', 'batch_size', 'path', 'debug']
        self.pca, self.pca_title, self.trainloader, self.testloader, self.testset, self.batch_size, self.path, self.debug = [args.get(key) for key in keys]

    def test(self, mode, image_type, text_type):
        path = os.path.join(self.path, mode)
        loader = self.testloader

        if not os.path.exists(path):
            os.makedirs(path)
        debug = self.debug

        pca = self.pca
        pca_title = self.pca_title

        reduced_images = []
        reduced_features_title = []
        labels = []
        if text_type != 'bert':
            for i,(images, targets, features_title) in tqdm(enumerate(loader), total=len(loader)):
                if len(targets) == self.batch_size:
                    pca_title.partial_fit(features_title.numpy())

        for i,(images, targets, features_title) in tqdm(enumerate(loader), total=len(loader)):
            reduced_images.append(np.squeeze(pca.transform(torch.flatten(images,1).numpy())))
            reduced_features_title.append(np.squeeze(pca_title.transform(features_title.numpy())))
            labels.append(targets.numpy())

        features = np.row_stack(reduced_images)
        labels = np.squeeze(np.concatenate(labels))
        features_title = np.row_stack(reduced_features_title)
            
        print('get ground_truth')
        ground_truth = ((labels[:,None]-labels[None,:]) == 0)+0

        m = len(labels)
         
        #get best component
        print('get best component')
        component_list = np.arange(1, 11) * 100
        component_title_list = np.arange(1, 11) * 50
        print(f'component_list: {component_list}')
        print(f'component_title_list: {component_title_list}')

        #image
        ap = np.zeros((len(component_list),m))
        roc = np.zeros((len(component_list),m))

        for j, component  in enumerate(component_list):
            dist = pdist(features[:,:component],metric='cosine')
            dist = squareform(dist)
            for i in tqdm(range(m)):
                score = 1 - dist[:,i:i+1]
                ap[j,i] = average_precision_score(ground_truth[:,i], score)
                roc[j,i] = roc_auc_score(ground_truth[:,i], score)
        ap = np.mean(ap, axis=1)
        roc = np.mean(roc, axis=1)
        print(f'ap: {ap}')
        print(f'roc: {roc}')

        #title
        ap_title = np.zeros((len(component_title_list),m))
        roc_title = np.zeros((len(component_title_list),m))

        for j, component_title  in enumerate(component_title_list):

            dist_title = pdist(features_title[:,:component_title],metric='cosine')
            dist_title = squareform(dist_title)

            for i in tqdm(range(m)):
                score_title = 1 - dist_title[:,i:i+1]
                ap_title[j,i] = average_precision_score(ground_truth[:,i], score_title)
                roc_title[j,i] = roc_auc_score(ground_truth[:,i], score_title)

        ap_title = np.mean(ap_title, axis=1)
        roc_title = np.mean(roc_title, axis=1)

        print(f'ap_title: {ap_title}')
        print(f'roc_title: {roc_title}')

        best_component = component_list[np.argmax((ap + roc) / 2)]
        best_component_title = component_title_list[np.argmax((ap_title + roc_title) / 2)]
            
        #get best threshold
        print('get best threshold')
        features = features[:,:best_component]
        features_title = features_title[:,:best_component_title]

        threshold_list = np.arange(1,10) * 0.1
        threshold_title_list = np.arange(1,10) * 0.1
        #threshold_list = [0.8, 0.9]
        #threshold_title_list = [0.8, 0.9]
        print(f'threshold_list: {threshold_list}')
        print(f'threshold_title_list: {threshold_title_list}')

        dist = pdist(features,metric='cosine')
        dist = squareform(dist)
        dist_title = pdist(features_title,metric='cosine')
        dist_title = squareform(dist_title)

        f1 = np.zeros((len(threshold_list),m))
        f1_title = np.zeros((len(threshold_title_list),m))

        for j, threshold in enumerate(threshold_list):
            for i in tqdm(range(m)):
                score = 1 - dist[:,i:i+1]
                pred = (score > threshold) 
                f1[j,i] = f1_score(ground_truth[:,i],pred)
        f1 = np.mean(f1, axis=1)
        print(f'f1: {f1}')
                
        for j, threshold_title in enumerate(threshold_title_list):
            for i in tqdm(range(m)):
                score_title = 1 - dist_title[:,i:i+1]
                pred_title = (score_title > threshold_title) 
                f1_title[j,i] = f1_score(ground_truth[:,i],pred_title)

        f1_title = np.mean(f1_title, axis=1)
        print(f'f1_title: {f1_title}')

        best_threshold = threshold_list[np.argmax(f1)]
        best_threshold_title = threshold_title_list[np.argmax(f1_title)]

        f1_merge = []
        for i in tqdm(range(m)):
            score = 1 - dist[:,i:i+1]
            pred = (score > best_threshold) 
            score_title = 1 - dist_title[:,i:i+1]
            pred_title = (score_title > best_threshold_title) 

            pred_merge = pred + pred_title
            f1_merge.append(f1_score(ground_truth[:,i],pred_merge))


        test_log = OrderedDict([('ap', ap[np.argmax((ap + roc) / 2)]), 
            ('roc', roc[np.argmax((ap + roc) / 2)]), 
            ('best_component', best_component),
            ('f1', np.max(f1)),
            ('threshold', best_threshold),
            ('ap_title', ap_title[np.argmax((ap_title + roc_title) / 2)]), 
            ('roc_title', roc_title[np.argmax((ap_title + roc_title) / 2)]), 
            ('best_component_title', best_component_title),
            ('f1_title', np.max(f1_title)),
            ('threshold_title', best_threshold_title),
            ('f1_merge', np.mean(f1_merge))])
        return test_log


    def train(self, image_type, text_type):
        pca = self.pca
        pca_title = self.pca_title
        for i,(images, labels, features_title) in tqdm(enumerate(self.trainloader), total=len(self.trainloader)):
            pca.partial_fit(images.numpy().reshape(images.shape[0],-1))
            if text_type == 'bert':
                pca_title.partial_fit(features_title.numpy())

        print(f'sum_variance_ratio: {np.sum(pca.explained_variance_ratio_)}')
        if text_type == 'bert':
            print(f'sum_variance_ratio: {np.sum(pca_title.explained_variance_ratio_)}')

class KNN_Matcher():
    def __init__(self, args):
        keys = ['knn', 'knn_title', 'trainloader', 'testloader', 'path', 'debug']
        self.knn, self.knn_title, self.trainloader, self.testloader, self.path, self.debug = [args.get(key) for key in keys]

    def test(self, mode, image_type, text_type):
        path = os.path.join(self.path, mode)
        loader = self.testloader
        if not os.path.exists(path):
            os.makedirs(path)

        _,(images, labels, features_title) = next(enumerate(loader))

        knn = self.knn
        knn_title = self.knn_title

        predictions = knn.predict(images.numpy().reshape((images.shape[0], -1)))
        predictions_title = knn_title.predict(features_title.numpy().reshape((features_title.shape[0], -1)))

        print('get ground_truth')
        ground_truth = ((labels[:,None]-labels[None,:]) == 0)+0
        f1 = []
        f1_title = []
        f1_merge = []

        m = len(labels)

        for i in tqdm(range(m)):
            pred = (predictions == predictions[i])
            f1.append(f1_score(ground_truth[:,i],pred))
            
            pred_title = (predictions_title == predictions_title[i])
            f1_title.append(f1_score(ground_truth[:,i],pred_title))
         
            pred_merge = pred + pred_title
            f1_merge.append(f1_score(ground_truth[:,i],pred_merge))

        test_log = OrderedDict([('f1', np.mean(f1)), 
            ('f1_title', np.mean(f1_title)), 
            ('f1_merge', np.mean(f1_merge))])
        return test_log


    def train(self, args):
        knn = self.knn
        knn_title = self.knn_title

        _, (images, labels, features_title) = next(enumerate(self.trainloader))
        knn.fit(images.numpy().reshape(images.shape[0],-1), labels.numpy())
        knn_title.fit(features_title.numpy(), labels.numpy())



 
if __name__ == '__main__':
    pass


















