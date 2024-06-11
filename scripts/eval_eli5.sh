#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=eval
#SBATCH --partition gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=15:00:00
#SBATCH --output=debug/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate selfrag
cd ~/rag-rerank

# Start the experiment.
python3 eval/eval.py --f result/eli5-Meta-Llama-3-8B-Instruct-extraction-shot0x0-ndoc5-42.json --citations --claims_nli
