
# BM25 100 + Vanilla 10
python3 crux-bm25-pointwise.py --default_config configs/crux/BM25_100-minilm_100.yaml \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.tsv \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    augmentation > logs/testb-BM25_100-minilm_100-vanilla_10.log 

# BM25 100 + Pointwise - minilm 100 + Vanilla 10
python3 crux-bm25-pointwise.py --default_config configs/crux/BM25_100-minilm_100.yaml \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.tsv \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    reranking \
    augmentation > logs/testb-BM25_100-minilm_100-vanilla_10.log 

# BM25 100 + Pointwise - monot5 100 + Vanilla 10
python3 crux-bm25-pointwise.py --default_config configs/crux/BM25_100-monot5_100.yaml \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.tsv \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    reranking \
    augmentation > logs/testb-BM25_100-monot5_100-vanilla_10.log 

# BM25 100 + Listwise - rankzephyr 100 + Vanilla 10
python3 crux-bm25-listwise.py --default_config configs/crux/BM25_100-rankgpt_100.yaml \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.tsv \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    reranking \
        --model_name_or_path castorini/rank_zephyr_7b_v1_full  \
    augmentation > logs/testb-BM25_100-rankzephyr_100-vanilla_10.log 

# BM25 100 + Listwise - rankfirst 100 + Vanilla 10
python3 crux-bm25-listwise.py --default_config configs/crux/BM25_100-rankfirst_100.yaml \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.tsv \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    retrieval \
    reranking \
        --model_name_or_path castorini/first_mistral \
        --use_logits \
    augmentation > logs/testb-BM25_100-rankfirst_100-vanilla_10.log 

