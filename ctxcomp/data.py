import torch
import random
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import (
    PaddingStrategy, 
    PreTrainedTokenizerBase
)

question_template = "Instruction: generate topic-related statements for each of the following documents in the given order. Ensure each statement is self-contained, concise and as short as possible. Topic: {} Documents:"
doc_template = "[{}] {}"
summary_template = "<pad> [{}]: {}{}"

@dataclass
class DataCollatorForContextCompressor:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    max_src_length: Optional[int] = 512
    max_tgt_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    n_contexts: Optional[int] = 1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        src, tgt = [], []
        for example in features:

            n = len(example['doc_ctxs'])
            random_idx = random.sample(range(n), n)[:self.n_contexts]
            ctx_orig, ctx_comp = [], ""
            for i, idx in enumerate(random_idx):
                ctx_orig += [doc_template.format(i+1, example['doc_ctxs'][idx])]
                if example['labels'][idx] == 1:
                    summaries = " ".join(example['comp_ctxs'][idx])
                    ctx_comp += summary_template.format(i+1, "", summaries)
                else:
                    ## irrelevant but still generate summary
                    summary = summary_template.format(i+1, "", example['comp_ctxs'][idx][0])
                    ctx_comp += summary

                    ## irrelevant with tag
                    # ctx_comp += "[{}]: irrelevant.".format(i+1)

            for j, _ in enumerate(range(max(self.n_contexts - n, 0))):
                ctx_orig += [""] # padding

            src += [question_template.format(example['question'])] 
            src += ctx_orig
            tgt.append(ctx_comp)

        inputs = self.tokenizer(
            src,
            max_length=self.max_src_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors='pt'
        )
        outputs = self.tokenizer(
            tgt,
            max_length=self.max_tgt_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors='pt'
        )
        target_mask = outputs['attention_mask'].bool()
        inputs['labels'] = outputs['input_ids'].masked_fill(~target_mask, -100)

        N = self.n_contexts + 1 
        inputs['input_ids'] = inputs['input_ids'].view(
            -1, N, inputs['input_ids'].size(-1)
        )
        inputs['attention_mask'] = inputs['attention_mask'].view(
            -1, N*inputs['attention_mask'].size(-1)
        )
        
        return inputs

