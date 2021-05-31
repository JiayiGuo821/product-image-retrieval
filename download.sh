#!/bin/bash
FILE=$1

if [ $FILE == "pretrained-network" ]; then
    URL=https://cloud.tsinghua.edu.cn/f/d4b8f2e4c67e4ef394aa/?dl=1
    mkdir -p ./expr/checkpoints/
    OUT_FILE=./expr/checkpoints/checkpoint_best.pth.tar
    wget -N $URL -O $OUT_FILE

elif  [ $FILE == "shopee-dataset" ]; then
    URL=https://cloud.tsinghua.edu.cn/f/5c7ba8c55e04478d86d9/?dl=1
    ZIP_FILE=./data/shopee.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE
    
elif  [ $FILE == "split-data" ]; then
    python split_data.py

else
    echo "Available arguments are pretrained-network, shopee-dataset and preprocess-data."
    exit 1

fi
