#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=summ
#SBATCH --partition gpu
#SBATCH --gres=gpu:tesla_p40:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=15:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/rag-rerank

# Start the experiment.

bart=facebook/bart-large-cnn
template='{T} {P}'
python3 recomp/summarize.py \
    --model_name_or_path ${bart} \
    --model_class seq2seq \
    --eval_file data/alce/eli5_eval_bm25_top100.json \
    --template '{T} {P}' \
    --batch_size 32 \
    --output_key summary_bart-large-cnn \
    --output_file eli5_eval_bm25_top100.json

