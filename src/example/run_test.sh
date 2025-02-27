
# bm25 100 + Vanilla 10
python3 crux-bm25-pointwise.py --default_config configs/crux/bm25_100-minilm_100.yaml \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/test_topics.tsv \
        --qrels_file /home/dju/datasets/crux/ranking_3/test_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/test_oracle-passages_judgements.jsonl \
        --n_questions 10 \
    retrieval \
    reranking \
    augmentation > logs/test-bm25_100-minilm_100-vanilla_10.log 

# bm25 100 + Pointwise - minilm 100 + Vanilla 10
python3 crux-bm25-pointwise.py --default_config configs/crux/bm25_100-minilm_100.yaml \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/test_topics.tsv \
        --qrels_file /home/dju/datasets/crux/ranking_3/test_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/test_oracle-passages_judgements.jsonl \
        --n_questions 10 \
    retrieval \
    reranking \
    augmentation > logs/test-bm25_100-minilm_100-vanilla_10.log 

# bm25 100 + Pointwise - monot5 100 + Vanilla 10
python3 crux-bm25-pointwise.py --default_config configs/crux/bm25_100-monot5_100.yaml \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/test_topics.tsv \
        --qrels_file /home/dju/datasets/crux/ranking_3/test_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/test_oracle-passages_judgements.jsonl \
        --n_questions 10 \
    retrieval \
    reranking \
    augmentation > logs/test-bm25_100-monot5_100-vanilla_10.log 

# bm25 100 + Listwise - rankzephyr 100 + Vanilla 10
python3 crux-bm25-listwise.py --default_config configs/crux/bm25_100-rankgpt_100.yaml \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/test_topics.tsv \
        --qrels_file /home/dju/datasets/crux/ranking_3/test_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/test_oracle-passages_judgements.jsonl \
        --n_questions 10 \
    retrieval \
    reranking \
        --model_name_or_path castorini/rank_zephyr_7b_v1_full  \
    augmentation > logs/test-bm25_100-rankzephyr_100-vanilla_10.log 

# bm25 100 + Listwise - rankfirst 100 + Vanilla 10
python3 crux-bm25-listwise.py --default_config configs/crux/bm25_100-rankfirst_100.yaml \
    data \
        --topic_file /home/dju/datasets/crux/ranking_3/test_topics.tsv \
        --qrels_file /home/dju/datasets/crux/ranking_3/test_qrels_pr.txt  \
        --judgement_file /home/dju/datasets/crux/ranking_3/test_oracle-passages_judgements.jsonl \
        --n_questions 10 \
    retrieval \
    reranking \
        --model_name_or_path castorini/first_mistral \
        --use_logits \
    augmentation > logs/test-bm25_100-rankfirst_100-vanilla_10.log 

