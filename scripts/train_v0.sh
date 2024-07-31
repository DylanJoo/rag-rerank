#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=std
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
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

# NUM_GPUS=4
# TOTAL_BATCH_SIZE=16
# BATCH_SIZE_PER_GPU=2
MODEL_DIR=/ivi/ilps/personal/dju/checkpoints

# GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --config_file configs/deepspeed_zero2_1gpu.yaml \
    ctxcomp/train.py \
    --model_name_or_path google/flan-t5-large \
    --tokenizer_name google/flan-t5-large \
    --config_name google/flan-t5-large  \
    --output_dir ${MODEL_DIR}/ctxcomp-flan-t5-arge-inverted-mds-std \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --train_file data/inverted-mds/test.jsonl \
    --max_src_length 768 \
    --max_tgt_length 512 \
    --optim adamw_torch \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant_with_warmup \
    --max_steps 10000 \
    --save_steps 1000 \
    --eval_steps 500 \
    --warmup_steps 500 \
    --do_train --do_eval \
    --bf16 true \
    --run_name std \
    --collator_type standard \
    --report_to wandb
