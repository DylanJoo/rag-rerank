. /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh  
conda activate rag
srun -c 8 --mem=32gb -p gpu --gres=gpu:1 --pty bash
# nvidia_rtx_a6000
