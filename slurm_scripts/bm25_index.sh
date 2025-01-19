#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=lucene.bm25
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag

# indexing
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input ${DATASET_DIR}/peS2o \
    --index ${INDEX_DIR}/peS2o/s2orc-v2.document.lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 32

