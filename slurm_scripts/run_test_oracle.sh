#!/bin/sh
#SBATCH --job-name=test-oracle_k
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
data=test

# Oracle retrieval + generation
for max_report_length in -1 1024; do

# relevance=3
python3 crux-oracle.py --default_config configs/crux/oracle_k.yaml \
    --exp ${data}-oracle_k-1:3_n${max_report_length} \
    data \
        --topic_file /home/dju/datasets/crux/ranking_5/${data}_topics.jsonl \
        --qrels_file /home/dju/datasets/crux/ranking_3/${data}_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/${data}_oracle-passages_judgements.jsonl \
        --threshold 3 \
        --n_questions 10 \
    retrieval \
    generation \
        --max_length $max_report_length \
    augmentation \
    --max_k -1 > logs/${data}-oracle_-1.log 

# relevance=2
python3 crux-oracle.py --default_config configs/crux/oracle_k.yaml \
    --exp ${data}-oracle_k-1:2_n${max_report_length} \
    data \
        --topic_file /home/dju/datasets/crux/ranking_5/${data}_topics.jsonl \
        --qrels_file /home/dju/datasets/crux/ranking_3/${data}_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/${data}_oracle-passages_judgements.jsonl \
        --threshold 2 \
        --n_questions 10 \
    retrieval \
    generation \
        --max_length $max_report_length \
    augmentation \
        --max_k 10 > logs/${data}-oracle_10.log 
done
