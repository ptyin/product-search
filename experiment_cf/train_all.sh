#!/bin/bash
models=("graph_search")
#datasets=("Automotive" "Cell_Phones_and_Accessories" "Clothing_Shoes_and_Jewelry" "Musical_Instruments" "Office_Products" "Toys_and_Games")
datasets=("Automotive" "Musical_Instruments" "Office_Products" "Toys_and_Games")
embedding_size=$1
convolution_num=$2
device=$3
cd ../src

for dataset in ${datasets[@]}; do
    for model in ${models[@]}; do
        if [ ! -d "/disk/yxk/log/cf/${embedding_size}/${dataset}/${model}/" ]; then
            mkdir -p "/disk/yxk/log/cf/${embedding_size}/${dataset}/${model}/"
        fi
        python main.py ${model} --convolution_num ${convolution_num} --embedding_size ${embedding_size} --dataset ${dataset} --device ${device} >> "/disk/yxk/log/cf/${embedding_size}/${dataset}/${model}/train_log_${convolution_num}.txt"
    done
done