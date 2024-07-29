from transformers import AutoTokenizer
from datasets import load_dataset
from data import DataCollatorForContextCompressor
from utils import update_tokenizer

ds = load_dataset('json', data_files='/home/dju/rag-rerank/data/inverted-mds/test.jsonl', keep_in_memory=True)['train']
batched = [ds[0], ds[1], ds[2] ]


tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
tokenizer = update_tokenizer(tokenizer)

d = DataCollatorForContextCompressor(
    tokenizer=tokenizer,
    padding=True,
    truncation=True,
    max_src_length=512,
    max_tgt_length=512,
    n_contexts=10
)

a = d(batched)

# print(a.input_ids.shape)
# print(a.attention_mask.shape)
# print(a.labels.shape)

## inputs
# b = a.input_ids[:, :, :100].contiguous()
# B, N, L = b.shape
#
# o = tokenizer.batch_decode(b.view(B, N*L), skip_special_tokens=True)
# print(o[0])
# print(o[1])
# print(o[2])


## outputs
b = a.labels
B, L = b.shape
b[b == -100] = 0

o = tokenizer.batch_decode(b, skip_special_tokens=False)
print(o[0])
print(o[1])


