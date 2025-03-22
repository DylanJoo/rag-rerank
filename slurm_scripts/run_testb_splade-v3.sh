#!/bin/sh
#SBATCH --job-name=testb-lsr
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
max_report_length=512
max_report_length=1024
max_report_length=-1
max_k=-1

# splade-v3 100 + Vanilla 10
python3 crux-splade-v3.py --default_config configs/crux/splade-v3_100-minilm_100.yaml \
    --exp testb-splade-v3_100-vanilla_${max_k} \
    data \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    generation \
        --max_length $max_report_length \
    augmentation \
        --max_k $max_k > logs/testb-splade-v3_100-vanilla_${max_k}.log 

# splade-v3 100 + Pointwise - minilm 100 + Vanilla 10
python3 crux-splade-v3.py --default_config configs/crux/splade-v3_100-minilm_100.yaml \
    --exp testb-splade-v3_100-minilm_100-vanilla_${max_k} \
    data \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    reranking \
    generation \
        --max_length $max_report_length \
    augmentation \
        --max_k $max_k > logs/testb-splade-v3_100-minilm_100-vanilla_${max_k}.log 

# splade-v3 100 + Pointwise - monot5 100 + Vanilla 10
python3 crux-splade-v3.py --default_config configs/crux/splade-v3_100-monot5_100.yaml \
    --exp testb-splade-v3_100-monot5_100-vanilla_${max_k} \
    data \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    reranking \
    generation \
        --max_length $max_report_length \
    augmentation \
        --max_k $max_k > logs/testb-splade-v3_100-monot5_100-vanilla_${max_k}.log 

# splade-v3 100 + Listwise - rankzephyr 100 (w20) + Vanilla 10
python3 crux-splade-v3.py --default_config configs/crux/splade-v3_100-rankgpt_100.yaml \
    --exp testb-splade-v3_100-rankzephyr_100-vanilla_${max_k} \
    data \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    listwise_reranking \
        --model_name_or_path castorini/rank_zephyr_7b_v1_full  \
    generation \
        --max_length $max_report_length \
    augmentation \
        --max_k $max_k > logs/testb-splade-v3_100-rankzephyr_100-vanilla_${max_k}.log 

# splade-v3 100 + Listwise - rankfirst 100 (w20) + Vanilla 10
python3 crux-splade-v3.py --default_config configs/crux/splade-v3_100-rankfirst_100.yaml \
    --exp testb-splade-v3_100-rankfirst_100-vanilla_${max_k} \
    data \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    listwise_reranking \
        --model_name_or_path castorini/first_mistral \
        --use_logits \
        --use_alpha \
    generation \
        --max_length $max_report_length \
    augmentation \
        --max_k $max_k > logs/testb-splade-v3_100-rankfirst_100-vanilla_${max_k}.log 

# splade-v3 100 + Setwise - 100 (w20) + Vanilla 10
python3 crux-splade-v3.py --default_config configs/crux/splade-v3_100-setwise_100.yaml \
    --exp testb-splade-v3_100-setwise_100-vanilla_${max_k} \
    data \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    listwise_reranking \
        --model_name_or_path google/flan-t5-xl \
        --type setwise \
    generation \
        --max_length $max_report_length \
    augmentation \
        --max_k $max_k > logs/testb-splade-v3_100-setwise_100-vanilla_${max_k}.log 

# splade-v3 100 + Pointwise - minilm 100 - mmr 10 + Vanilla 10
python3 crux-splade-v3.py --default_config configs/crux/splade-v3_100-minilm_100-mmr_10.yaml \
    --exp testb-splade-v3_100-minilm_100-mmr_10-vanilla_${max_k} \
    data \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    reranking \
    listwise_reranking \
    generation \
        --max_length $max_report_length \
    augmentation \
        --max_k $max_k > logs/testb-splade-v3_100-minilm_100-mmr_10-vanilla_${max_k}.log 
