# Product Image Retrieval

Final Project for Course Pattern Recognition Spring 2021, Tsinghua Univ.


## Software installation

**Assume you have [Git](https://git-scm.com/downloads) and [Anaconda](https://www.anaconda.com/products/individual) installed yet.**

Clone this repository:

```
git clone https://github.com/JiayiGuo821/product-image-retrieval.git
cd product-image-retrieval/
```

or if you are the TA, just unzip our submitted zip file and run:
```
cd codes/
```

Install the dependencies:
```
conda create -n product-image-retrieval python=3.6.7
conda activate product-image-retrieval
pip install -r requirements.txt
```
## Dataset and pretrained network
We provide a script to download dataset and pre-trained network to reproduce our best results. The splitted data and network checkpoint will be downloaded and stored in the data/splitted and expr/checkpoints directories, respectively.

```
bash download.sh shopee-dataset
bash download.sh split-data
bash download.sh pretrained-network
```

## Reproduce our best results


### Training

#### MoCo v2 
Assume you have at least 4 GPUs.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_moco.py \
  -a resnet50 \
  --lr 0.015 \
  --batch-size 128 \
  --dist-url 'tcp://localhost:20001' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --moco-t 0.2 --aug-plus --cos -j 16\
  --moco-dim 1000 --moco-k 4096\
  ./data
```


### Evaluation

Get image features (MoCo v2) from the best checkpoint we provide:
```
python test_moco.py -a resnet50 --batch-size 64 --mlp -j 16 -s test --pretrained ./expr/checkpoints/checkpoint_best.pth.tar --moco-dim 1000 --name best ./data
```
or from a checkpoint you train:
```
python test_moco.py -a resnet50 --batch-size 64 --mlp -j 16 -s test --pretrained ./expr/checkpoints/checkpoint_0024.pth.tar --moco-dim 1000 --name best ./data
```
Get text features (Tfidf) and the final metric result:
```
python test_predict.py --name moco_best --image_type features_mocobest_test --text_type tfidf --mode test
```


## Results

### Results on Text

| Model | F1 score | AUC | mAP |
|---|---|---|---|
| Tfidf | 0.7555 | 0.9931 | 0.8718 |

### Results on Image

| Model | F1 score | AUC | mAP |
|---|---|---|---|
| MoCo v2 | 0.7482 | 0.9923 | 0.8441 |

### Results on Image & Text

| Model | F1 score | AUC | mAP |
|---|---|---|---|
| MoCo v2 + Tfidf | 0.8095 | - | - |


## Effects of files not mentioned above
```
test_bert.py # compute text features with bert
train_knn.py # train knn
train_pca.py # train pca
test_predict_knn.py # reproduce knn-related results in our report
test_predict_pca.py # reproduce pca-related results in our report
datasets.py, models.py, moco\ & nets\ # some utils 
```


## Contacts
{guo-jy20, dcq20}@mails.tsinghua.edu.cn

