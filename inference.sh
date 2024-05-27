#!/bin/sh
#SBATCH --job-name=INFER
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_xp:2
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env
export CUDA_HOME=/usr/local/cuda-11

cd src/
python inference.py
