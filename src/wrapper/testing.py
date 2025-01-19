
# index_dir='/home/dju/indexes/peS2o/s2orc-v2.document.lucene'
# corpus_dir='/home/dju/datasets/peS2o/'
index_dir='/home/dju/indexes/litsearch/bm25.litsearch.full_documents.lucene/'
corpus_dir='/home/dju/datasets/litsearch/full_paper/corpus.jsonl'
example_topic = {'1': 'Are there any research papers on methods to compress large-scale language models using task-agnostic knowledge distillation techniques?'}
# golden doc: 202719327
writer = None

""" I. First-stage Retrieval
## [TODO] query reformulation 
"""
from retrieve.bm25 import search
output_run = search(
    index=index_dir,
    k1=0.9,
    b=0.4,
    topics=example_topic,
    batch_size=4,
    k=1000, 
    writer=writer
)

""" II. Retrieval Augmentation 
## [TODO] Add listwise (mulitdocument) reranking/summarization
"""
from tools.ranking_utils import load_corpus
corpus = load_corpus(corpus_dir)

## II(a). Passage reranking
from augment.pointwise import rerank
output_run = rerank(
    topics=example_topic,
    corpus=corpus,
    runs=output_run,
    reranker_config={
        "reranker_class": 'monobert',
        "reranker_name_or_path": 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        "device": 'cuda',
        "fp16": True
    },
    top_k=1000,
    batch_size=2,
    max_length=512,
    writer=writer,
)

## II(b). Passage summariation
from augment.pointwise import summarize
output_context = summarize(
    topics=example_topic,
    corpus=corpus,
    runs=output_run,
    summarizer_config={
        "summarizer_class": 'seq2seq',
        "summarizer_name_or_path": 'google/flan-t5-base',
        'fp16': True,
        'flash_attention_2': False
    },
    top_k=10,
    batch_size=2,
    max_length=1024,
    template="Summarize the document based on the query. Query: {q} Document: {d} Summary: ",
    writer=writer,
)
print(output_context)


""" III. Generation
## [TODO] Huggingface generation pipeline (maybe dont need)
"""

# from augment.pointwise import filter

# Generate
# from generate.llm.vllm_back import vLLM
# generator = vLLM(model=model_opt.generator_name_or_path, temperature=0.7)
