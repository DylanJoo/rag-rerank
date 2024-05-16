import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple
from transformers import TrainingArguments

@dataclass
class ModelOptions:
    model_debug: Optional[bool] = field(default=False)
    model_name_or_path: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    use_flash_attn: Optional[bool] = field(default=None)
    use_special_tokens: bool = field(default=False)
    use_slow_tokenizer: bool = field(default=False)

@dataclass
class DataOptions:
    train_file: Optional[str] = field(default=None)
    eval_data_dir: Optional[str] = field(default=None)
    max_seq_length: Optional[int] = field(default=512)

@dataclass
class TrainOptions(TrainingArguments):
    # output_dir: str = field(default='./')
    # do_train: bool = field(default=False)
    # do_eval: bool = field(default=False)
    overwrite_output_dir: bool = field(default=True)
    wandb_project: Optional[str] = field(default=None)

    # lora training
    use_lora: Optional[bool] = field(default=False)
    lora_rank: Optional[int] = field(default=64)
    lora_alpha: Optional[float] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)

    # optimizer
    weight_decay: float = field(default=0.0)
    learning_rate: Union[float] = field(default=5e-5)
    gradient_accumulation_steps: Optional[int] = field(default=16)
    gradient_checkpointing: Optional[bool] = field(default=True)
    lr_scheduler_type: Optional[str] = field(default="linear")
    warmup_ratio: Optional[float] = field(default=0)
    with_tracking: Optional[bool] = field(default=False)

    # training setup
    low_cpu_mem_usage: Optional[bool] = field(default=False)
    max_steps: int = field(default=None) # different from HF's
    # num_train_epochs: int = field(default=3)
    # save_strategy: str = field(default='epoch')
    # save_steps: int = field(default=1000)
    # eval_steps: int = field(default=1000)
    # evaluation_strategy: Optional[str] = field(default='no')
    # per_device_train_batch_size: int = field(default=2)
    # per_device_eval_batch_size: int = field(default=2)

    # dataloader_num_workers: Optional[int] = field(default=4)
    # logging_dir: Optional[str] = field(default='./logs')
    # remove_unused_columns: bool = field(default=False)
