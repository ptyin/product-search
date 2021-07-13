#!/bin/bash

cd ../preprocess/

#datasets=("Automotive" "Cell_Phones_and_Accessories" "Clothing_Shoes_and_Jewelry" "Musical_Instruments" "Office_Products" "Toys_and_Games")
dataset=$1
#max_users_list=(5 3 1)
bought_nums=(2 3 4 5)

# user cold start
for bought_num in ${bought_nums[@]}; do
#  for dataset in ${datasets[@]}; do
    python preprocess.py --bought_num ${bought_num} --dataset "${dataset}" --processed_path "/disk/yxk/processed/cold_start/"
    python doc2vec.py --dataset "${dataset}" --processed_path "/disk/yxk/processed/cold_start/${dataset}/${bought_num}/"
    python transform.py --csv_dir "/disk/yxk/processed/cold_start/${dataset}/${bought_num}/" --review_file "/disk/yxk/data/cold_start/reviews_${dataset}_5.json.gz" --output_dir "/disk/yxk/transformed/cold_start/${dataset}/${bought_num}" --meta_file "/disk/yxk/data/cold_start/meta_${dataset}.json.gz" --dataset ${dataset}
#  done
done
