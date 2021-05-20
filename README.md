# Product Image Retrieval

Final Project for Course Pattern Recognition Spring 2021, Tsinghua Univ.


## Software installation

**Assume you have [Git](https://git-scm.com/downloads) and [Anaconda](https://www.anaconda.com/products/individual) installed yet.**

Clone this repository:

```
git clone https://github.com/JiayiGuo821/product-image-retrieval.git
cd product-image-retrieval/
```
Install the dependencies:
```
conda create -n product-image-retrieval python=3.6.7
conda activate product-image-retrieval
pip3 install -r requirement.txt
```
## Dataset and pretrained network
We provide a script to download dataset and pre-trained network to reproduce our best results. The splitted data and network checkpoint will be downloaded and stored in the data/splitted and expr/checkpoints directories, respectively.

```
bash download.sh shopee-dataset
bash download.sh split-data
bash download.sh pretrained-network # not available yet
```

## Usage


### Training
As an example, use the following command to train a CondenseNetV2-A/B/C on ImageNet

```
python -m torch.distributed.launch --nproc_per_node=8 train.py --model cdnv2_a/b/c 
  --batch-size 1024 --lr 0.4 --warmup-lr 0.1 --warmup-epochs 5 --opt sgd --sched cosine \
  --epochs 350 --weight-decay 4e-5 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 \
  --data_url /PATH/TO/IMAGENET --train_url /PATH/TO/LOG_DIR
```


### Evaluation
We take the ImageNet model trained above as an example.

To evaluate the non-converted trained model, use `test.py` to evaluate from a given checkpoint path:

```
python test.py --model cdnv2_a/b/c \
  --data_url /PATH/TO/IMAGENET -b 32 -j 8 \
  --train_url /PATH/TO/LOG_DIR \
  --evaluate_from /PATH/TO/MODEL_WEIGHT
```

To evaluate the converted trained model, use `--model converted_cdnv2_a/b/c`:

```
python test.py --model converted_cdnv2_a/b/c \
  --data_url /PATH/TO/IMAGENET -b 32 -j 8 \
  --train_url /PATH/TO/LOG_DIR \
  --evaluate_from /PATH/TO/MODEL_WEIGHT
```

Note that these models are still the large models after training. To convert the model to standard group-convolution version as described in the paper, use the `convert_and_eval.py`:

```
python convert_and_eval.py --model cdnv2_a/b/c \
  --data_url /PATH/TO/IMAGENET -b 64 -j 8 \
  --train_url /PATH/TO/LOG_DIR \
  --convert_from /PATH/TO/MODEL_WEIGHT
```

## Results

### Results on ImageNet

| Model | FLOPs | Params | Top-1 Error | Tsinghua Cloud | Google Drive |
|---|---|---|---|---|---|
| CondenseNetV2-A | 46M | 2.0M | 35.6 | [Download](https://cloud.tsinghua.edu.cn/smart-link/34933e0e-565b-4633-b1ea-a5266d3d3fcc/) | [Download](https://drive.google.com/file/d/1fhHeAGkdZnOEgv9f-IUCy_uNfc-QHcZ_/view?usp=sharing) |
| CondenseNetV2-B | 146M | 3.6M | 28.1 | [Download](https://cloud.tsinghua.edu.cn/smart-link/444627eb-a296-458e-9a44-db38aca8a761/) | [Download](https://drive.google.com/file/d/1xFR3GcV1tsGq4tHhPS50XCW7AMnfWs6E/view?usp=sharing) |
| CondenseNetV2-C | 309M | 6.1M | 24.1 | [Download](https://cloud.tsinghua.edu.cn/smart-link/4625ac39-54b2-48c1-bcbd-c6d21a6b42fa/) | [Download](https://drive.google.com/file/d/1QaK-5KtVeK-d6ip8RMJhJ87dVmPAnWEA/view?usp=sharing) |

### Results on COCO2017 Detection
The detection experiments are conducted based on the [mmdetection repository](https://github.com/open-mmlab/mmdetection). We simply replace the backbones of FasterRCNN and RetinaNet with our CondenseNetV2s.

| Detection Framework | Backbone | Backbone FLOPs | mAP |
|---|---|---|---|
| FasterRCNN | ShuffleNetV2 0.5x | 41M | 22.1 |
| FasterRCNN | CondenseNetV2-A | 46M | 23.5 |
| FasterRCNN | ShuffleNetV2 1.0x | 146M | 27.4 |
| FasterRCNN | CondenseNetV2-B | 146M | 27.9 |
| FasterRCNN | MobileNet 1.0x | 300M | 30.6 |
| FasterRCNN | ShuffleNetV2 1.5x | 299M | 30.2 |
| FasterRCNN | CondenseNetV2-C | 309M | 31.4 |
| RetinaNet  | MobileNet 1.0x | 300M | 29.7 |
| RetinaNet  | ShuffleNetV2 1.5x | 299M | 29.1 |
| RetinaNet  | CondenseNetV2-C | 309M | 31.7 |

### Results on CIFAR

| Model | FLOPs | Params | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|
| CondenseNet-50 | 28.6M | 0.22M | 6.22 | - |
| CondenseNet-74 | 51.9M | 0.41M | 5.28 | - |
| CondenseNet-86 | 65.8M | 0.52M | 5.06 | 23.64 |
| CondenseNet-98 | 81.3M | 0.65M | 4.83 | - |
| CondenseNet-110 | 98.2M | 0.79M | 4.63 | - |
| CondenseNet-122 | 116.7M | 0.95M | 4.48 | - |
| CondenseNetV2-110 | 41M | 0.48M | 4.65 | 23.94 |
| CondenseNetV2-146 | 62M | 0.78M | **4.35** | **22.52** |

## Contacts
{guo-jy20, dcq20}@mails.tsinghua.edu.cn

