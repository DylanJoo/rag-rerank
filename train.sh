#!/bin/sh
#SBATCH --job-name=RAG
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:4
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env
cd src/

# Start the experiment.
MODEL_SIZE=1.1B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=4
MODEL_DIR=/ivi/ilps/personal/dju/checkpoints

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file stage3_no_offloading_accelerate.conf \
    finetune_hf.py \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v0.6 \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-Chat-v0.6 \
    --use_slow_tokenizer \
    --train_file ~/datasets/selfrag/train.small.jsonl \
    --max_seq_length 16 \
    --dataloader_num_workers 32 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --gradient_checkpointing \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --output_dir ${MODEL_DIR}/self_rag_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --use_special_tokens