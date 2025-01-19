import torch
import json
import re
import os
import string
from glob import glob
from collections import defaultdict

def update_tokenizer(tokenizer, max_n_contexts=10):
    tokenizer.add_special_tokens({"additional_special_tokens": ["<cls>"]})
    return tokenizer

def load_model(
    summarizer_name_or_path, 
    summarizer_class='causal', 
    dtype='fp16', 
    load_mode=None,
):

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    from retrieval_augmentation.models import FiDT5
    from llm.base import vLLM

    model_cls_map = {"fid": FiDT5, "seq2seq": AutoModelForSeq2SeqLM, "causal": vLLM}[summarizer_class.lower()]

    tokenizer = AutoTokenizer.from_pretrained(summarizer_name_or_path, use_fast=False)

    if (summarizer_class == 'fid') or (summarizer_class == 'seq2seq'):
        model = model_cls_map.from_pretrained(summarizer_name_or_path).to('cuda')
        model.eval()

    if summarizer_class == 'vllm':
        from types import SimpleNamespace
        args_dict = {'model': summarizer_name_or_path, 'num_gpus': 1, 'quant': None, 'top_p': 0.95, 'temperature': 0.7}
        args = SimpleNamespace(**args_dict)
        model = vLLM(args)

    return model, tokenizer

def load_judgements(path):
    judgements = defaultdict(lambda: defaultdict(lambda: None))
    contexts = defaultdict(lambda: None)
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                example_id = data['example_id']
                judgements[example_id].update({data['pid']: data['rating']})
                if 'contents' in data.keys():
                    contexts[example_id].update({data['pid']: data['contents']})
    return judgements, contexts

