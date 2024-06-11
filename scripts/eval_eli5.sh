#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=eval
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=debug/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate selfrag
cd ~/rag-rerank

# Start the experiment.
# for file in result/eli5-Meta-Llama-3-8B-Instruct-extraction-*.json;do
#     python3 eval/eval.py --f ${file} --citations --claims_nli
# done
file=result/eli5-Meta-Llama-3-8B-Instruct-extraction-shot0x0-ndoc5-42-quick_test10.json
python eval/eval.py --f ${file} --citations --claims_nli
