#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=autoalign
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag

cd ~/rag-rerank

python3 auto_alignment.py \
    --input_dir data/mds/ \
    --multi_news_file /home/dju/datasets/multi_news \
    --wcep_10_file /home/dju/datasets/wcep-10 \
    --dataset_name mds \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --temperature 1.0 \
    --load_mode vllm \
    --top_p 0.95 \
    --tag align-gen \
    --max_new_tokens 10 --quick_test 5000 \
    --shot 0 --ndoc_in_demo 0 --ndoc 0  \
    --ampere_gpu
