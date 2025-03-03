# This script is for direct  
python3 -m evaluation.llmjudge.retrieval_augmentation_generation \
    --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.jsonl \
    --corpus_dir /home/dju/datasets/crux/passages/ \
    --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt \
    --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --threshold 3 \
    --gamma 0.5  \
    --output_jsonl testing.jsonl
