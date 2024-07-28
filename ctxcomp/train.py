import sys
import multiprocessing
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig
)
from datasets import load_dataset

# customized modules
from data import DataCollatorForStarter
from trainers import TrainerForStarter
from models import FiDT5
from arguments import ModelOpt, DataOpt, TrainOpt

import os

def main():
    # Parse argument for huggingface packages
    parser = HfArgumentParser((ModelArgs, DataArgs, Seq2SeqTrainArgs))
    model_opt, data_opt, training_opt = parser.parse_args_into_dataclasses()

    # Model
    tokenizer = AutoTokenizer.from_pretrained(model_opt.tokenizer_name)
    model = FiDT5.from_pretrained(model_opt.model_name_or_path)

    # Dataset
    dataset = load_dataset('json', data_files=data_opt.train_file, keep_in_memory=True)
    n_examples = len(dataset['train'])
    if training_opt.do_eval:
        dataset = dataset['train'].train_test_split(test_size=100, seed=1997)

    # Datacollator
    data_collator = DataCollatorForContextCompressor(
        retrieval_enhanced=model_opt.retrieval_enhanced,
        tokenizer=tokenizer, 
        max_src_length=data_opt.max_p_length,
        max_tgt_length=data_opt.max_q_length,
        truncation=True,
        padding=True,
        sep_token='</s>',
    )

    # Trainer
    trainer = Trainer(
        model=model, 
        args=training_opt,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
    )
    results = trainer.train()

if __name__ == '__main__':
    main()
