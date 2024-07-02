#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=naive
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:2
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/rag-rerank

# Start the experiment.
SUMMARY=data/add_summary/eli5_eval_bm25_top100.json

## zero-shot
python3 naive.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 30 --shot 0 --ndoc_in_demo 0 --ndoc 5 --used_field text
python3 naive.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 30 --shot 0 --ndoc_in_demo 0 --ndoc 5 --used_field summary
python3 naive.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 30 --shot 0 --ndoc_in_demo 0 --ndoc 5 --used_field summary_bart-large-cnn --eval_file $SUMMARY --used_field_in_demo summary 
python3 naive.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 30 --shot 0 --ndoc_in_demo 0 --ndoc 10 --used_field text
python3 naive.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 30 --shot 0 --ndoc_in_demo 0 --ndoc 10 --used_field summary
python3 naive.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 30 --shot 0 --ndoc_in_demo 0 --ndoc 10 --used_field summary_bart-large-cnn --eval_file $SUMMARY --used_field_in_demo summary 

## one-shot ICL
python3 naive.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 30 --shot 1 --ndoc_in_demo 5 --ndoc 5 --used_field text
python3 naive.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 30 --shot 1 --ndoc_in_demo 5 --ndoc 5 --used_field summary
python3 naive.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 30 --shot 1 --ndoc_in_demo 5 --ndoc 5 --used_field summary_bart-large-cnn  --eval_file $SUMMARY --used_field_in_demo summary 
python3 naive.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 30 --shot 1 --ndoc_in_demo 5 --ndoc 10 --used_field text
python3 naive.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 30 --shot 1 --ndoc_in_demo 5 --ndoc 10 --used_field summary
python3 naive.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 30 --shot 1 --ndoc_in_demo 5 --ndoc 10 --used_field summary_bart-large-cnn  --eval_file $SUMMARY --used_field_in_demo summary 

