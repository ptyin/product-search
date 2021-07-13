#!/bin/bash
model=$1
#datasets=("Automotive" "Cell_Phones_and_Accessories" "Clothing_Shoes_and_Jewelry" "Musical_Instruments" "Office_Products" "Toys_and_Games")
datasets=("Office_Products" "Toys_and_Games")
device=$2
cd ../src

for dataset in ${datasets[@]}; do
    if [ ! -d "/disk/yxk/log/cf/64/${dataset}/${model}/" ]; then
        mkdir -p "/disk/yxk/log/cf/64/${dataset}/${model}/"
    fi
    python main.py ${model} --dataset ${dataset} --device ${device} >> "/disk/yxk/log/cf/64/${dataset}/${model}/train_log.txt"
done