#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=oracle
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:2
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=debug/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate selfrag
cd ~/rag-rerank

# Start the experiment.
# python3 llm/oracle.py --config configs/eli5.tinyllama-1b-chat.0shot.oracle-top5.yaml
# python3 llm/oracle.py --config configs/eli5.tinyllama-1b-chat.1shot.oracle-top5.yaml
#
# python3 llm/oracle.py --config configs/eli5.llama-8b-chat.0shot.oracle-top5.yaml --load_mode 4int
python3 llm/oracle.py --config configs/eli5.llama-8b-chat.1shot.oracle-top5.yaml --load_mode 4int
