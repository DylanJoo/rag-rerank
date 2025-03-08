#!/bin/sh
#SBATCH --job-name=testb-dr
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
# contriever 100 + Vanilla 10
python3 crux-contriever.py --default_config configs/crux/contriever_100-minilm_100.yaml \
    --exp testb-contriever_100-vanilla_10 --online_eval \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.jsonl \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    generation \
    augmentation > logs/testb-contriever_100-vanilla_10.log 

# contriever 100 + Pointwise - minilm 100 + Vanilla 10
python3 crux-contriever.py --default_config configs/crux/contriever_100-minilm_100.yaml \
    --exp testb-contriever_100-minilm_100-vanilla_10 --online_eval \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.jsonl \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    reranking \
    generation \
    augmentation > logs/testb-contriever_100-minilm_100-vanilla_10.log 

# contriever 100 + Pointwise - monot5 100 + Vanilla 10
python3 crux-contriever.py --default_config configs/crux/contriever_100-monot5_100.yaml \
    --exp testb-contriever_100-monot5_100-vanilla_10 --online_eval \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.jsonl \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    reranking \
    generation \
    augmentation > logs/testb-contriever_100-monot5_100-vanilla_10.log 

# contriever 100 + Listwise - rankzephyr 100 (w20) + Vanilla 10
python3 crux-contriever.py --default_config configs/crux/contriever_100-rankgpt_100.yaml \
    --exp testb-contriever_100-rankzephyr_100-vanilla_10 --online_eval \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.jsonl \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    listwise_reranking \
        --model_name_or_path castorini/rank_zephyr_7b_v1_full  \
    generation \
    augmentation > logs/testb-contriever_100-rankzephyr_100-vanilla_10.log 

# contriever 100 + Listwise - rankfirst 100 (w20) + Vanilla 10
python3 crux-contriever.py --default_config configs/crux/contriever_100-rankfirst_100.yaml \
    --exp testb-contriever_100-rankfirst_100-vanilla_10 --online_eval \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.jsonl \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    listwise_reranking \
        --model_name_or_path castorini/first_mistral \
        --use_logits \
        --use_alpha \
    generation \
    augmentation > logs/testb-contriever_100-rankfirst_100-vanilla_10.log 

# contriever 100 + Setwise - 100 (w20) + Vanilla 10
python3 crux-contriever.py --default_config configs/crux/contriever_100-setwise_100.yaml \
    --exp testb-contriever_100-setwise_100-vanilla_10 --online_eval \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.jsonl \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    listwise_reranking \
        --model_name_or_path google/flan-t5-xl \
        --type setwise \
    generation \
    augmentation > logs/testb-contriever_100-setwise_100-vanilla_10.log 

# contriever 100 + Pointwise - minilm 100 - mmr 10 + Vanilla 10
python3 crux-contriever.py --default_config configs/crux/contriever_100-minilm_100-mmr_10.yaml \
    --exp testb-contriever_100-minilm_100-mmr_10-vanilla_10 --online_eval \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.jsonl \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    reranking \
    listwise_reranking \
    generation \
    augmentation > logs/testb-contriever_100-minilm_100-mmr_10-vanilla_10.log 
