#!/bin/bash
set -e

models=("distilbert" "roberta" "gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl")
datasets=("imdb" "edos" "sst-2")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Downloading ${model} with dataset ${dataset} from Huggingface..."
        output_path="../Sparsify-then-Classify/model"
        mkdir -p "$(dirname "$output_path")"
        python3 initialize.py --model_name ${model} --dataset ${dataset}
    done
done
