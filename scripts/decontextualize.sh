#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=decont
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
cd ~/rag-rerank

# Start the experiment.

## zero-shot
python3 decontextualize.py \
    --config configs/mds-decontextualize.llama3-8b-chat.yaml \
    --quick_test 5000 --shot 0 --ndoc_in_demo 0 --ndoc 0 

## one-shot ICL
