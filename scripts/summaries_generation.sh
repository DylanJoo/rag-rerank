#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=summgen
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/rag-rerank

# Start the experiment.

# for shard_i in $(seq 0 10);do
for shard_i in $(seq 11 24);do
    python3 decontextualize_summs.py \
        --shard $shard_i --shard_size 200 \
        --config configs/mds-decontextualize.llama3-8b-chat.yaml \
        --tag summ-gen \
        --load_mode no \
        --temperature 0.7 \
        --max_new_tokens 768 \
        --quick_test 5000 \
        --ampere_gpu 
done
