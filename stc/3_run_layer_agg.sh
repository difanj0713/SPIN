#!/bin/bash
set -e

models=("distilbert") # "roberta" "gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl")
datasets=("imdb") # "edos" "sst-2")
finetuned=0

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Layer-wise Aggregation and Classification for model ${model} over dataset ${dataset}..."
        output_path="../dataset_acts/${dataset}/new_agg"
        mkdir -p ${output_path}
        python3 layer_agg.py --model_name ${model} --dataset ${dataset} --is_finetuned ${finetuned}
    done
done