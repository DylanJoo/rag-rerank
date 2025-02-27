#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=lucene.bm25
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ~/.bashrc
. /home/dju/miniconda3/etc/profile.d/conda.sh
conda activate rag

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

python -m pyserini.index.lucene \
    --collection BeirFlatCollection \
    --input ${DATA_DIR}/litsearch/abstracts \
    --index ${INDEX_DIR}/litsearch/bm25.litsearch.abstracts.lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 128 

# CRUX corpus (which is default pyserini corpus)
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input ${DATA_DIR}/crux/passages \
    --index ${INDEX_DIR}/crux/bm25.crux.passages.lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 128

python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input ${DATA_DIR}/crux/documents \
    --index ${INDEX_DIR}/crux/bm25.crux.documents.lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 128
