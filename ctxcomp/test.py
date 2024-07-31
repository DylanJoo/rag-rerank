from transformers import AutoTokenizer
from datasets import load_dataset
from utils import update_tokenizer

ds = load_dataset('json', data_files='/home/dju/rag-rerank/data/inverted-mds/test.jsonl', keep_in_memory=True)['train']
batched = [ds[0], ds[1], ds[2] ]
batched = [ds[0] ]


tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
tokenizer = update_tokenizer(tokenizer)

from data import Standard, StandardWithPrefix
d = StandardWithPrefix(
    tokenizer=tokenizer,
    max_src_length=1025,
    max_tgt_length=512,
    n_contexts=None
)

a = d(batched)

# print(a.input_ids.shape)
# print(a.attention_mask.shape)
# print(a.labels.shape)

# inputs
b = a.input_ids[:, :, :100].contiguous()
B, N, L = b.shape

o = tokenizer.batch_decode(b.view(B, N*L), skip_special_tokens=False)
print(o[0])
print('label')
a.labels[a.labels == -100] = 0
print(tokenizer.decode(a.labels[0], skip_special_tokens=True))
print(tokenizer.decode(a.labels[1], skip_special_tokens=True))

# from models import FiDT5
# from models.fidt5_comp import FiDT5
# from transformers import AutoTokenizer
#
# MODEL_PATH='google/flan-t5-large'
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# from utils import update_tokenizer
# tokenizer = update_tokenizer(tokenizer)
# MODEL_PATH='/ivi/ilps/personal/dju/checkpoints/ctxcomp-flan-t5-arge-inverted-mds/checkpoint-1000'
# model = FiDT5.from_pretrained(MODEL_PATH)
#
# output = model.generate(
#     input_ids=a.input_ids, 
#     attention_mask=a.attention_mask,
#     do_sample=True,
#     top_p=0.1,
#     temperature=0.2,
#     min_new_tokens=256,
#     max_new_tokens=512,
# )

# print(tokenizer.decode(output[0], skip_special_tokens=False))
# print(tokenizer.decode(output[1], skip_special_tokens=True))
# print(tokenizer.decode(output[2], skip_special_tokens=True))

