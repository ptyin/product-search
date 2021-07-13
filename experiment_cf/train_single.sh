#!/bin/bash
#models=("lse" "hem" "aem" "zam" "tran_search" "graph_search")
model=$1
#datasets=("Automotive" "Cell_Phones_and_Accessories" "Clothing_Shoes_and_Jewelry" "Musical_Instruments" "Office_Products" "Toys_and_Games")
dataset=$2
embedding_size=64
#convolution_num=$3
device=$3
cd ../src

if [ ! -d "/disk/yxk/log/cf/${embedding_size}/${dataset}/${model}/" ]; then
    mkdir -p "/disk/yxk/log/cf/${embedding_size}/${dataset}/${model}/"
fi
#python main.py ${model} --save_str ${model}_${convolution_num} --convolution_num ${convolution_num} --embedding_size ${embedding_size} --dataset ${dataset} --device ${device} >> "/disk/yxk/log/cf/${embedding_size}/${dataset}/${model}/train_log_${convolution_num}.txt"
python main.py ${model} --save_str no_id_prop --mode no_id_prop --embedding_size ${embedding_size} --dataset ${dataset} --device ${device} >> "/disk/yxk/log/cf/${embedding_size}/${dataset}/${model}/train_log_no_id_prop.txt"