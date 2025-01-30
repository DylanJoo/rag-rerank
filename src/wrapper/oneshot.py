index_dir='/home/dju/indexes/litsearch/bm25.litsearch.full_documents.lucene/'
corpus_dir='/home/dju/datasets/litsearch/corpus.jsonl'
example_topic = {
    '1': 'Are there any research papers on methods to compress large-scale language models using task-agnostic knowledge distillation techniques?',
    '2': 'Are there any research papers on new language model pre-training that can outperform masked language modeling?'
}

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
    k=100, 
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
    top_k=100,
    batch_size=2,
    max_length=512,
)

## II(b). Passage filtering
### [TODO] use the reranking model as the selection baseline.
from augment.pointwise import select
output_context = select(
    topics=example_topic,
    corpus=corpus,
    runs=output_run,
    selector_config={
        "reranker_class": 'monobert',
        "reranker_name_or_path": 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        "device": 'cuda',
        "fp16": True
    },
    max_k=3,
    batch_size=2,
    max_length=512,
    threshold=0.0,
)


""" III. Generation
"""
PROMPT = "Write a passage that answers the given query. Use the provided search results to draft the answer (some of them might be irrelevant). Cite the documents if they are relevant. Write the passage within 100 words. Add the `<p>` and `</p>` tags at the beginning and the end.\n\nQuery: {Q}\nSearch results:\n{Ds}\nPassage: <p>"

# from generate.llm.vllm_back import vLLM
from generate.llm.hf_back import LLM
generator = LLM(model='meta-llama/Llama-3.2-1B-Instruct', temperature=0.7)
xs = []
for qid in example_topic:
    q = example_topic[qid]
    ds = output_context[qid]
    xs.append(PROMPT.replace("{Q}", q).replace("{Ds}", ds))

output_response = generator.generate(x=xs, max_tokens=500)
print([r.split('</p>')[0] for r in output_response])
