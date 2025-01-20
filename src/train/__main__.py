"""
[Deprecated codes] 
if train_opt.use_lora:
    from peft import get_peft_model, LoraConfig
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'],
        modules_to_save=['cross_attn']
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import sys
import torch
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig,
    Trainer
)
from datasets import load_dataset

# customized modules
from train.options import ModelOpt, DataOpt, TrainOpt
from train.models import FiDT5, LlamaForCausalContextLM # put modeling here so far. After training done, intergrate it to the pipeline.
from utils import update_tokenizer, load_model

class VerboseTrainer(Trainer):

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        if (self.state.global_step % 10 == 0) and (self.args.should_save):
            pass
            # kwargs = {
            #     'input_ids': inputs.input_ids[:1],
            #     'attention_mask': inputs.attention_mask[:1],
            #     'max_new_tokens': 512,
            #     'temperature': 0.7
            # }
            # if 'decoder_input_ids' in inputs:
            #     kwargs.update({'decoder_input_ids': inputs.decoder_input_ids[:1]})
            #
            # with torch.no_grad():
            #     output = model.generate(**kwargs)
            # logger.info(self.tokenizer.decode(output[0], skip_special_tokens=False))

        return super().compute_loss(
            model=model,
            inputs=inputs,
        )

def load_model(model_name_or_path, model_class='causualLM', device='cpu'):
    from models import FiDT5, LlamaForCausalContextLM
    MODEL_CLASS = {"fid": FiDT5, "cepe": LlamaForCausalContextLM}[model_class]
    if model_class == 'cepe':
        model_kwargs = {
            "attn_implementation": "flash_attention_2" if device == 'cuda' else "eager",
            "torch_dtype": torch.bfloat16
        }
    else:

    model = MODEL_CLASS.from_pretrained(
        model_name_or_path, 
        device_map=device,
        **model_kwargs
    )
    model.train_encoder = False

    return model

def main():
    parser = HfArgumentParser((ModelOpt, DataOpt, TrainOpt))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()

    # Model
    if 't5' in model_opt.model_name_or_path:
        model = FiDT5.from_pretrained(
            model_opt.model_name_or_path, 
            device_map='auto',
            attn_implementation="sdpa", 
            torch_dtype=torch.float16
        )
    if 'llama' in model_opt.model_name_or_path:
        model = LlamaForCausalContextLM.from_pretrained(
            model_opt.model_name_or_path, 
            device_map='auto',
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16
        )
    tokenizer = AutoTokenizer.from_pretrained(model_opt.model_name_or_path)

    ## resizing embeddings (adding new tokens is recommended (instead of preserved tokens))
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Dataset
    dataset = load_dataset('json', data_files=data_opt.train_file, keep_in_memory=True)
    dataset = dataset.filter(lambda x: len(x['docids']) !=0 )
    n_examples = len(dataset['train'])
    if train_opt.do_eval:
        dataset = dataset['train'].train_test_split(test_size=100, seed=1997)

    # Datacollator
    from data.collator import Standard, StandardWithPrefix
    collator_class = {
        "standard": Standard, 
        "standard_with_prefix": StandardWithPrefix
    }[train_opt.collator_type]

    data_collator = collator_class(
        tokenizer=tokenizer, 
        max_src_length=data_opt.max_src_length,
        max_tgt_length=data_opt.max_tgt_length,
        max_num_contexts=train_opt.max_num_contexts,
        num_distractor_docs=train_opt.num_distractor_docs,
        num_redundant_docs=train_opt.num_redundant_docs,
        shuffle=True,
    )
    # num_contexts is for batch-wsie training

    # Trainer
    train_opt.gradient_checkpointing_kwargs={"use_reentrant": True}
    trainer = VerboseTrainer(
        model=model, 
        args=train_opt,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if train_opt.do_eval else None,
        data_collator=data_collator,
    )
    results = trainer.train()

if __name__ == '__main__':
    main()
