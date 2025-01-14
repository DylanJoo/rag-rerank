#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=bm25
#SBATCH --cpus-per-task=32
#SBATCH --nodes=2
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag

DATASET_DIR=/home/dju/datasets/peS2o
INDEX_DIR=/home/dju/indexes/peS2o

# index
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input ${DATASET_DIR}/documents \
    --index ${INDEX_DIR}/bm25.peS2o.documents.lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 128

