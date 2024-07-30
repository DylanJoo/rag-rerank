#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=traincc1
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
export WANDB_PROJECT=rak-ctxcomp-flant5
source ${HOME}/.bashrc
conda activate rag

cd ~/rag-rerank

NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=16
MODEL_DIR=/ivi/ilps/personal/dju/checkpoints

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch --config_file configs/deepspeed_zero2.yaml \
    ctxcomp/train.py \
    --model_name_or_path google/flan-t5-large \
    --tokenizer_name google/flan-t5-large \
    --config_name google/flan-t5-large  \
    --output_dir ${MODEL_DIR}/ctxcomp-flan-t5-arge-inverted-mds \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --train_file data/inverted-mds/test.jsonl \
    --max_src_length 512 \
    --max_tgt_length 512 \
    --n_contexts 5 \
    --optim adafactor \
    --learning_rate 3e-5 \
    --lr_scheduler_type constant_with_warmup \
    --max_steps 5000 \
    --save_steps 1000 \
    --eval_steps 500 \
    --do_train --do_eval \
    --bf16 true \
    --run_name cc-query_focuse\
    --report_to wandb
