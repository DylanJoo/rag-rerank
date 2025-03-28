#!/bin/sh
#SBATCH --job-name=testb-random
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

# Start experiments
# max_report_length=512
# max_report_length=1024
max_report_length=-1
max_k=10
ADD_GENERATION=true

mkdir -p logs/rac_${max_k}

# BM25 50 + random 100 Vanilla 10
# for seed in $(seq 1 16); do
# python3 crux-random.py --default_config configs/crux/bm25_100-minilm_100.yaml \
#     --exp testb-bm25_50-random_${seed}-vanilla_${max_k} \
#     data \
#         --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
#         --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
#     retrieval --k 50 --seed ${seed} \
#     ${ADD_GENERATION:+generation --max_length ${max_report_length}} \
#     augmentation \
#         --max_k $max_k > logs/rac_${max_k}/testb-bm25_50-random_${seed}-vanilla_${max_k}.log
# done

# contriever 50 + random 100 Vanilla 10
for seed in $(seq 1 16); do
python3 crux-random.py --default_config configs/crux/contriever_100-minilm_100.yaml \
    --exp testb-contriever_50-random_${seed}-vanilla_${max_k} \
    data \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval --k 50 --seed ${seed} \
    ${ADD_GENERATION:+generation --max_length ${max_report_length}} \
    augmentation \
        --max_k $max_k > logs/rac_${max_k}/testb-contriever_50-random_${seed}-vanilla_${max_k}.log
done

# splade 50 + random 100 Vanilla 10
for seed in $(seq 1 16); do
python3 crux-random.py --default_config configs/crux/splade-v3_100-minilm_100.yaml \
    --exp testb-splade-v3_50-random_${seed}-vanilla_${max_k} \
    data \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval --k 50 --seed ${seed} \
    ${ADD_GENERATION:+generation --max_length ${max_report_length}} \
    augmentation \
        --max_k $max_k > logs/rac_${max_k}/testb-splade-v3_50-random_${seed}-vanilla_${max_k}.log
done
