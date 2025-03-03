#!/bin/sh
#SBATCH --job-name=lucene.splade
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ~/.bashrc
enter_conda
conda activate rag

python3 -m src.retrieve.mlm_encode \
    --model_name_or_path naver/splade-v3 \
    --tokenizer_name naver/splade-v3 \
    --collection ${DATA_DIR}/crux/passages \
    --collection_output ${INDEX_DIR}/crux/splade-v3.crux.passages.lucene/encoded/vectors.jsonl \
    --batch_size 128 \
    --max_length 512 \
    --quantization_factor 100
