from tools import load_corpus, load_topics

# Data
index_dir='/home/dju/indexes/crux/bm25.crux.passages.lucene/'
corpus_dir='/home/dju/datasets/crux/passages/'
topic_file='/home/dju/datasets/crux/ranking_3/test_topics.tsv'
all_topic = load_topics(topic_file)
example_topic = {"1": all_topic[next(iter(all_topic))]}

""" I. First-stage Retrieval """
from retrieve.bm25 import search
output_run = search(
    index=index_dir,
    k1=0.9,
    b=0.4,
    topics=example_topic,
    batch_size=4,
    k=1000, 
)

""" II. Retrieval Augmentation """
corpus = load_corpus(corpus_dir)

## II(a). Passage reranking (mono)
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
    top_k=30,
    batch_size=2,
    max_length=1024,
    template="Summarize the document based on the query. Query: {q} Document: {d} Summary: ",
)


""" III. Generation """
PROMPT = "Write a passage that answers the given query. Use the provided search results to draft the answer (some of them might be irrelevant). Cite the documents if they are relevant. Write the passage within 100 words. Add the `<p>` and `</p>` tags at the beginning and the end.\n\nQuery: {Q}\nSearch results:\n{Ds}\nPassage: <p>"

# from generate.llm.vllm_back import vLLM
from generate.llm.hf_back import LLM
generator = LLM(model='meta-llama/Llama-3.2-1B-Instruct', temperature=0.7)
xs = []
for qid in example_topic:
    q = example_topic[qid]
    ds = output_context[qid]['contexts']
    xs.append(PROMPT.replace("{Q}", q).replace("{Ds}", ds))
output_response = generator.generate(x=xs, max_tokens=500)

for i, qid in enumerate(example_topic):
    print("############")
    print("# Request: \n", example_topic[qid], '\n')
    print("# Output: \n", output_response[0].split('</p>'))
    print("############")

""" IV. Evaluation """

