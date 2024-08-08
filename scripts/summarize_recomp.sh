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

recomp=fangyuan/nq_abstractive_compressor
template='Question: {Q}\nDocument: {T} {P}\nSummary: '
python3 ctxcompt/summarize.py \
    --model_name_or_path ${recomp} \
    --model_class seq2seq \
    --eval_file data/alce/eli5_eval_bm25_top100.json \
    --template ${template} \
    --batch_size 32 \
    --truncate \
    --output_key summary_recomp-nq \
    --output_file eli5_eval_bm25_top100.json

