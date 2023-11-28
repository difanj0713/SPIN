#!/bin/bash
set -e

models=("distilbert") # "roberta" "gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl")
datasets=("imdb") # "edos" "sst-2")
finetuned=0

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Getting Activations and Hidden States from ${model} with dataset ${dataset} from Huggingface..."
        output_path="../Sparsify-then-Classify/dataset_acts"
        mkdir -p "$(dirname "$output_path")"
        python3 initialize.py --model_name ${model} --dataset ${dataset} --is_finefuned ${finetuned}
    done
done
