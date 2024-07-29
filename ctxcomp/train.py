import sys
import multiprocessing
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig,
    Seq2SeqTrainer
)
from datasets import load_dataset

# customized modules
from data import DataCollatorForContextCompressor
from models import FiDT5
from options import ModelOpt, DataOpt, TrainOpt
from utils import update_tokenizer

import os

def main():
    # Parse argument for huggingface packages
    parser = HfArgumentParser((ModelOpt, DataOpt, TrainOpt))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()

    # Model
    tokenizer = AutoTokenizer.from_pretrained(model_opt.tokenizer_name)
    model = FiDT5.from_pretrained(model_opt.model_name_or_path)
    model.use_cache = False
    tokenizer = update_tokenizer(tokenizer)

    ## resizing embeddings (adding new tokens is recommended (instead of preserved tokens))
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Dataset
    dataset = load_dataset('json', data_files=data_opt.train_file, keep_in_memory=True)
    n_examples = len(dataset['train'])
    if train_opt.do_eval:
        dataset = dataset['train'].train_test_split(test_size=100, seed=1997)

    # Datacollator
    data_collator = DataCollatorForContextCompressor(
        tokenizer=tokenizer, 
        max_src_length=data_opt.max_p_length,
        max_tgt_length=data_opt.max_q_length,
        padding=True,
        truncation=True,
        n_contexts=train_opt.n_contexts
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model, 
        args=train_opt,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if train_opt.do_eval else None,
        data_collator=data_collator,
    )
    results = trainer.train()

if __name__ == '__main__':
    main()
