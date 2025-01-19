import json
import string
import random
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer
)

class dummyLLM:

    def generate(self, inputs):
        if len(inputs[0]) > 512:
            outputs = []
            for input in inputs:
                strings = input.split()
                random.shuffle(strings)
                outputs.append( " ".join(strings[:100]) )
            return outputs
        else:
            return inputs

class LLM:

    def __init__(self, model, temperature=0.7, top_p=0.9, flash_attention_2=False):
        self.load_model(model, flash_attention_2=flash_attention_2)
        self.temperature = temperature
        self.top_p = top_p

    def load_model(self, model_name_or_path, dtype=torch.float16, flash_attention_2=False):

        if flash_attention_2:
            model_kwargs = {
                'torch_dtype': torch.bfloat16,
                'attn_implementation': "flash_attention_2"
            }
        else:
            model_kwargs = {'torch_dtype': dtype}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto',
            **model_kwargs
        )
        self.model.eval()

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        self.tokenizer.padding_side = "left"

    def generate(self, x, max_tokens=1024, min_tokens=0, **kwargs):

        if isinstance(x, str):
            x = [x]

        inputs = self.tokenizer(x, padding=True, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=self.temperature, 
            top_p=self.top_p, 
            min_new_tokens=min_tokens,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
        )
        generation = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        del inputs, outputs
        torch.cuda.empty_cache()
        return generation

class Seq2seqLLM(LLM):

    def load_model(self, model_name_or_path, dtype=torch.float16, flash_attention_2=False):

        if flash_attention_2:
            model_kwargs = {
                'torch_dtype': torch.bfloat16,
                'attn_implementation': "flash_attention_2"
            }
        else:
            model_kwargs = {'torch_dtype': dtype}

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            device_map='auto',
            **model_kwargs
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    def generate(self, x, max_tokens=1024, min_tokens=0, **kwargs):

        if isinstance(x, str):
            x = [x]

        inputs = self.tokenizer(x, padding=True, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**inputs, min_new_tokens=min_tokens, max_new_tokens=max_tokens)
        generation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        del inputs, outputs
        torch.cuda.empty_cache()
        return generation

