#!/bin/sh
#SBATCH --job-name=testb-oracle_k
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ~/.bashrc
enter_conda
conda activate rag

# root
cd src 

# Oracle retrieval + generation
for max_report_length in -1 512 1024; do
python3 crux-oracle.py --default_config configs/crux/oracle_k.yaml \
    --exp testb-oracle_k \
    data \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    generation \
        --max_length $max_report_length \
    augmentation \
        --max_k -1 > logs/testb-oracle_k.log 
done
