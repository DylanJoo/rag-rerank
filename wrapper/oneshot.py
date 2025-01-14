import argparse

# def main(args):
#     pass

# Retrieval
from retrieve.bm25 import search
## Retrieval -- query reformulation

writer = open('temp.run', 'w')
output = search(
    index='/home/dju/indexes/s2orc-v2.document.lucene', 
    k1=0.9,
    b=0.4,
    topics={'1': 'What is bi-directional attention.'},
    batch_size=4,
    k=10, 
    writer=writer
)
writer.close()

# Augment
from augment.pointwise import rerank

from augment.pointwise import filter

# Generate
# from generate.llm.vllm_back import vLLM
# generator = vLLM(model=model_opt.generator_name_or_path, temperature=0.7)
