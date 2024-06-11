#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=oracle
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
python3 oracle.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 10 --ndoc 5 --shot 0
python3 oracle.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 10 --ndoc 5 --shot 1 --ndoc_in_demo 5
python3 oracle.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 10 --ndoc 5 --shot 2 --ndoc_in_demo 5
python3 oracle.py --config configs/eli5.llama3-8b-chat.yaml --quick_test 10 --ndoc 5 --shot 3 --ndoc_in_demo 3
