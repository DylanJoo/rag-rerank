#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=eval
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/rag-rerank

# Start the experiment.
# Instruct-oracle
for file in result/*oracle*json;do
    python3 eval/eval.py --f ${file} --citations --claims_nli
done
