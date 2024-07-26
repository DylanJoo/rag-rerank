#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=flat-index
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag

cd ~/rag-rerank

# flatten generated doc-claims
python3 data_augmentation/flatten.py \
    --input_dir data/mds  \
    --output_dir data/mds/doc_claims

# indexing
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input data/mds/doc_claims \
    --index ${INDEX_DIR}/mds-doc-claims \
    --generator DefaultLuceneDocumentGenerator \
    --threads 8 \
    --storePositions --storeDocvectors --storeRaw 

