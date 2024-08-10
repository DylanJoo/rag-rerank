# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag

cd ~/rag-rerank

# construct pairs
python3 data_augmentation/compose_pairs.py \
    --input_dir data/mds  \
    --dataset_jsonl data/inverted-mds/test2.jsonl  \
    --alignment_jsonl data/mds/alignment/question_to_summaries_llama3.1.jsonl \
    --threshold 3 \
    --n_max_docs 5 \
    --n_max_distractors 5 \
    --doc_passages_index /home/dju/indexes/mds-docs-psgs/  \
    --good_read
