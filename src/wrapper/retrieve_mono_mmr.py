from tools import load_corpus, load_topics, load_qrels, load_judgements

# Data
index_dir='/home/dju/indexes/crux/bm25.crux.passages.lucene/'
corpus_dir='/home/dju/datasets/crux/passages/'
topics_file='/home/dju/datasets/crux/ranking_3/testb_topics.tsv'
qrels_file='/home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt'
judgements_file='/home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl'

example_topic = load_topics(topics_file)
corpus = load_corpus(corpus_dir)
# example_topic = {"1": all_topic[next(iter(all_topic))]}
qrels = load_qrels(qrels_file)
judgements = load_judgements(judgements_file)

""" I. First-stage Retrieval """
from retrieve.bm25 import search
output_runs = search(
    index=index_dir,
    k1=0.9,
    b=0.4,
    topics=example_topic,
    batch_size=4,
    k=100, 
)

""" II. Retrieval Augmentation """
## II(a). Passage reranking (mono)
from augment.pointwise import rerank
output_runs = rerank(
    topics=example_topic,
    corpus=corpus,
    runs=output_runs,
    reranker_config={
        "reranker_class": 'monobert',
        "reranker_name_or_path": 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        "device": 'cuda',
        "fp16": True
    },
    top_k=None,
    batch_size=64,
    max_length=512,
)

## II(b). MMR reranking
# from augment.selection import mmr_rerank
# output_runs = mmr_rerank(
#     topics=example_topic,
#     encoder_name_or_path='sentence-transformers/all-mpnet-base-v2',
#     corpus=corpus,
#     runs=output_runs,
#     max_k=10,
#     batch_size=64,
#     lambda_param=0.9,
#     max_length=512,
# )


## II(c). Context augmentation
from augment.base import vanilla
output_rac = vanilla(
    topics=example_topic,
    corpus=corpus,
    runs=output_runs,
    max_k=10,
)

""" III. Intermediate Evaluation """
from evaluation import rac_evaluate
output_eval = rac_evaluate(
    corpus=corpus,
    qrels=qrels, 
    judgements=judgements,
    rac_data=output_rac,
    n_questions=15,
    threshold=3,
    # runs=output_run,
)
print(output_eval)


""" IV. Generation """
PROMPT = "Write a passage that answers the given query. Use the provided search results to draft the answer (some of them might be irrelevant). Cite the documents if they are relevant. Write the passage within 100 words. Add the `<p>` and `</p>` tags at the beginning and the end.\n\nQuery: {Q}\nSearch results:\n{Ds}\nPassage: <p>"

# from generate.llm.vllm_back import vLLM
# from generate.llm.hf_back import LLM
# generator = LLM(model='meta-llama/Llama-3.2-1B-Instruct', temperature=0.7)
# xs = []
# for qid in example_topic:
#     q = example_topic[qid]
#     ds = output_context[qid]['prompt']
#     xs.append(PROMPT.replace("{Q}", q).replace("{Ds}", ds))
# output_response = generator.generate(x=xs, max_tokens=500)
#
# for i, qid in enumerate(example_topic):
#     print("############")
#     print("# Request: \n", example_topic[qid], '\n')
#     print("# Output: \n", output_response[0].split('</p>'))
#     print("############")

""" VI. Evaluation """
