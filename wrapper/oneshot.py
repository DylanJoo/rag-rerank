import argparse

# def main(args):
#     pass

# Retrieval
## Load retrieval modules
from retrieve.bm25 import search
## Retrieval -- query reformulation
## Retrieval -- search

writer = open('temp.run', 'w')
output = search(
    index='/home/dju/indexes/litsearch.bm25_lucene_doc',
    k1=0.9,
    b=0.4,
    topics={'1': 'What is bi-directional attention.'},
    batch_size=4,
    k=10, 
    writer=writer
)
writer.close()

# Generate
from generate.llm.base import vLLM
generator = vLLM(model=model_opt.generator_name_or_path, temperature=0.7)
