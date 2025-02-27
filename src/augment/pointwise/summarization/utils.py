import torch
import json
import re
import os
import string
from glob import glob
from collections import defaultdict
from generate.llm import vllm_back, hf_back 

def load_summarizer(
    model_class='seq2seq', 
    model_name_or_path='t5-base', 
    temperature=0.7,
    top_p=0.95,
    **kwargs
):

    model_cls_map = {"seq2seq": hf_back.LLM, "causal": vllm_back.LLM}[model_class.lower()]

    if model_class == 'seq2seq':
        model = model_cls_map(
            model=model_name_or_path,
            flash_attention_2=kwargs.pop('flash_attention_2', False)
        )

    if model_class == 'vllm':
        model = model_cls_map(
            model=model_name_or_path, 
            temperature=temperature, 
            top_p=top_p,
            num_gpus=kwargs.get("num_gpus")
        )

    return model

def truncate_and_concat(texts, tokenizer, max_length=512, offset=6):
    tokenized = tokenizer.tokenize(texts)
    length = len(tokenizer.tokenize(texts))
    max_length = (max_length or tokenizer.max_len_single_sentence-1)
    if (length+offset) < max_length:
        return texts
    else:
        return tokenizer.convert_tokens_to_string(tokenized[:(max_length-6)])

def update_tokenizer(tokenizer, max_n_contexts=10):
    tokenizer.add_special_tokens({"additional_special_tokens": ["<cls>"]})
    return tokenizer
