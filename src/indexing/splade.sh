#!/bin/sh
#SBATCH --job-name=splade
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

RETRIEVER=naver/splade-v3
DATASET_DIR=/home/dju/datasets/peS2o
INDEX_DIR=/home/dju/indexes/peS2o
    # --index ${INDEX_DIR}/bm25.documents.lucene \

# Encode 
python -m index.mlm_encode \
    --model_name_or_path ${RETRIEVER} \
    --tokenizer_name ${RETRIEVER} \
    --collection_dir ${DATASET_DIR}/documents \
    --collection_output ${INDEX_DIR}/splade-v3.peS2o.documents.encoded/vectors.jsonl \
    --batch_size 256 \
    --max_length 256 \
    --quantization_factor 100

# Index (cpu only)
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ${INDEX_DIR}/splade-v3.peS2o.documents.encoded \
  --index ${INDEX_DIR}/splade-v3.peS2o.documents.lucene \
  --generator DefaultLuceneDocumentGenerator \
  --threads 36 \
  --impact --pretokenized
