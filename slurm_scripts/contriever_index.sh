#!/bin/sh
#SBATCH --job-name=faiss.ctr
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ~/.bashrc
enter_conda
conda activate rag

python -m pyserini.encode \
    input   --corpus ${DATA_DIR}/crux/passages \
            --fields text \
            --delimiter "__IMPOSSIBLE_DELIMITER__" \
    output  --embeddings ${INDEX_DIR}/crux/contriever.crux.passages.faiss \
            --to-faiss \
    encoder --encoder facebook/contriever-msmarco \
            --encoder-class contriever \
            --fields text \
            --max-length 512 \
            --batch 64 \
            --fp16

# [TODO] adjust them to dense retrieval
# peS2o
# python -m pyserini.index.lucene \
#     --collection JsonCollection \
#     --input ${DATA_DIR}/peS2o \
#     --index ${INDEX_DIR}/peS2o/bm25.s2orc-v2.document.lucene \
#     --generator DefaultLuceneDocumentGenerator \
#     --threads 32

# Litsearch corpus (transformed to BEIR-style corpus)
# see collection here. https://github.com/castorini/anserini/tree/master/src/main/java/io/anserini/collection
# python -m pyserini.index.lucene \
#     --collection BeirFlatCollection \
#     --input ${DATA_DIR}/litsearch/full_paper \
#     --index ${INDEX_DIR}/litsearch/bm25.litsearch.full_documents.lucene \
#     --generator DefaultLuceneDocumentGenerator \
#     --threads 128 

