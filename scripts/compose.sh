# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag

cd ~/rag-rerank

# construct pairs
python3 data_augmentation/compose_pairs.py \
    --input_dir data/mds  \
    --result_jsonl data/mds/alignment/anchor_to_nuggets_TRUE_bm25.jsonl \
    --dataset_jsonl data/inverted-mds/test.jsonl  \
    --n_claims_per_query 10 \
    --min_scores -1 \
    --n_max_docs 10

    # --result_jsonl data/mds/alignment/summary_claims_alignment_bm25top1.jsonl \
# compose into the ctxcomp tasks
# python3 data_augmentation/compose_pairs.py \
#     --input_dir data/mds/archived  \
#     --result_jsonl data/mds/alignment/nugget_claims_alignment_bm25top1.jsonl \
#     --dataset_jsonl test/dataset_train.v2.jsonl  \
#     --n_claims_per_query 2 \
#     --min_scores -10 \
#     --n_max_docs 5
