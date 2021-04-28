#!/bin/bash
models=("lse" "hem" "aem" "zam" "tran_search" "graph_search")
datasets=("Automotive" "Cell_Phones_and_Accessories" "Clothing_Shoes_and_Jewelry" "Musical_Instruments" "Office_Products" "Toys_and_Games")
embedding_size=$1
device=$2
cd ../src

for dataset in ${datasets[@]}; do
    for model in ${models[@]}; do
        if [ ! -d "/disk/yxk/log/cf/${embedding_size}/${dataset}/${model}/" ]; then
            mkdir -p "/disk/yxk/log/cf/${embedding_size}/${dataset}/${model}/"
        fi
        python main.py ${model} --embedding_size ${embedding_size} --dataset ${dataset} --device ${device} >> "/disk/yxk/log/cf/${embedding_size}/${dataset}/${model}/train_log.txt"
    done
done