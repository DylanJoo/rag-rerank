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
enter_conda
conda activate rag

python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ${INDEX_DIR}/crux/splade-v3.crux.passages.lucene/encoded \
  --index ${INDEX_DIR}/crux/splade-v3.crux.passages.lucene \
  --generator DefaultLuceneDocumentGenerator \
  --threads 36 \
  --storeDocvectors --impact --pretokenized

