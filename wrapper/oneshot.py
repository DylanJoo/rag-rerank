import argparse
from tools.ranking_utils import load_corpus

# index_dir='/home/dju/indexes/peS2o/s2orc-v2.document.lucene'
# corpus_dir='/home/dju/datasets/peS2o/'
index_dir='/home/dju/indexes/litsearch/bm25.litsearch.full_documents.lucene/'
corpus_dir='/home/dju/datasets/litsearch/full_paper/corpus.jsonl'

# Retrieval
## [TODO] query reformulation
from retrieve.bm25 import search
writer = open('temp.run', 'w')
output_run = search(
    index=index_dir,
    k1=0.9,
    b=0.4,
    topics={'1': 'Are there any research papers on methods to compress large-scale language models using task-agnostic knowledge distillation techniques?'},
    batch_size=4,
    k=1000, 
    writer=writer
)
writer.close()
# print('[RETRIEVAL]', output_run)

# Augment
corpus = load_corpus(corpus_dir)
from augment.pointwise import rerank
output_run = rerank(
    topics={'1': 'Are there any research papers on methods to compress large-scale language models using task-agnostic knowledge distillation techniques?'},
    corpus=corpus,
    runs=output_run,
    reranker_config={
        "reranker_class": 'monobert',
        "reranker_name_or_path": 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'device': 'cuda',
        'fp16': True
    },
    writer=None,
    top_k=1000,
    batch_size=2,
    max_length=512
)
# print('[RERANKING]', output_run)

# from augment.pointwise import filter

# Generate
# from generate.llm.vllm_back import vLLM
# generator = vLLM(model=model_opt.generator_name_or_path, temperature=0.7)
