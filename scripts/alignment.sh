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

# option1. basic. full-summary to document's claims
# python3 data_augmentation/align_pseudo_entailment.py \
#     --input_dir data/mds  \
#     --result_jsonl data/mds/alignment/summary_claims_alignment_bm25top1.jsonl \
#     --doc_claims_index ${INDEX_DIR}/mds-doc-claims \
#     --full_nuggets_as_premise

# option1. basic. full-summary to document's claims
python3 data_augmentation/align_pseudo_entailment.py \
    --input_dir data/mds  \
    --result_jsonl data/mds/alignment/nugget_claims_alignment_bm25top1.jsonl \
    --doc_claims_index ${INDEX_DIR}/mds-doc-claims 

