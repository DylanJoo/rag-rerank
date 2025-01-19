import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

class Seq2seq(nn.Module):

    def __init__(
        self, 
        model_name_or_dir='t5-base',
        tokenizer_name=None, 
        device='auto', 
        fp16=False
    ):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_dir,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            device_map=device
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name_or_dir
        )

        self.input_template = "{}"

    def summarize(self, queries, documents, titles=None, max_length=512):
        # [NOTE] sometimes the token `relevant` token will be truncated
        if titles is not None:
            documents = [f'{title} {text}'.strip() for title, text in zip(titles, documents)] 
        text_pairs = [f"Query: {q} Document: {d} Relevant:" for (q, d) in zip(queries, documents)]
        inputs = self.tokenizer(
            text_pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_length
        ).to(self.model.device)

        # generate first token 
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2,
            return_dict_in_generate=True,
            output_scores=True
        )

        # scoring the first logits and apply softmax
        scores = outputs.scores[0]
        scores = scores[:, [self.NREL, self.REL]]
        scores = F.log_softmax(scores, dim=-1)
        scores = scores[:, 1].exp().cpu().detach().numpy()
        return scores

    def __str__(self):
        return self.model.name_or_path
