#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=align
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

# construct pairs
python3 data_augmentation/construct_pairs.py \
    --input_dir data/mds  \
    --result_jsonl data/mds/alignment/summary_claims_alignment_bm25top1.jsonl.sample \
    --output_jsonl data/inverted-mds/test.jsonl  \
    --n_max_claims 30 \
    --n_max_docs 5

# compose into the ctxcomp tasks
