#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=statgen
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
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

## zero-shot
for shard_i in $(seq 0 24);do
    python3 decontextualize.py \
        --shard $shard_i --shard_size 200 \
        --config configs/mds-decontextualize.llama3-8b-chat.yaml \
        --tag stat-gen \
        --max_new_tokens 512 --quick_test 5000 \
        --shot 0 --ndoc_in_demo 0 --ndoc 0  \
        --ampere_gpu 
done

# python3 decontextualize.py \
# --shard $shard_i --shard_size 200 \
# --config configs/mds-decontextualize.llama3-8b-chat.yaml \
# --tag claims-gen --generation claims \
# --max_new_tokens 512 --quick_test 5000 --shot 0 --ndoc_in_demo 0 --ndoc 0 \
# --ampere_gpu 
