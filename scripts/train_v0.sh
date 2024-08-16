#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=std
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
export WANDB_PROJECT=rak-ctxcomp-flant5
source ${HOME}/.bashrc
conda activate rag
export CUDA_HOME=/usr/local/cuda

cd ~/rag-rerank

MODEL_DIR=/ivi/ilps/personal/dju/checkpoints

# GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --config_file configs/deepspeed_zero2_2gpu.yaml \
    ctxcomp/train.py \
    --model_name_or_path google/flan-t5-large \
    --tokenizer_name google/flan-t5-large \
    --config_name google/flan-t5-large  \
    --output_dir ${MODEL_DIR}/ctxcomp-v2-flan-t5-large-inverted-mds-std-310 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --train_file data/inverted-mds/mds-5k-greedy-1.jsonl \
    --max_src_length 1024 \
    --max_tgt_length 1024 \
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
    --max_num_contexts 3 \
    --num_distractor_docs 1 \
    --num_redundant_docs 0 \
    --collator_type standard \
    --report_to wandb

