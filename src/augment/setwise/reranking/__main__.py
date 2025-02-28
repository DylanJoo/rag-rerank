"""
The listwise ranking is mostly relying on `llm-rankers` repositary.
See details in `https://github.com/ielab/llm-rankers`
"""
import copy
from typing import Any, Dict, List, Union
from operator import itemgetter
from tqdm import tqdm

# [NOTE] these data formats are from `rank_llm` package
from rank_llm.data import Result, Query, Candidate 
from llmrankers.setwise import SetwiseLlmRanker
from llmrankers.rankers import SearchResult

def convert_runs_to_pairs(topics=None, corpus=None, runs=None, top_k=50):
    pairs = []
    for qid in runs:
        candidates = []
        for i, docid in enumerate(runs[qid]):
            if i < top_k:
                candidates.append(
                    SearchResult(docid=docid, score=runs[qid][docid], text=corpus[docid])
                )
        pairs.append((topics[qid], candidates))
    return pairs

# [NOTE] the original rerank function is not using batch processing
def rerank(
    topics, corpus, runs, 
    model_path: str,
    top_k: int = 100,
    writer=None,
    **kwargs
):
    """
    # [TODO] add vllm backend for generation
    # [NOTE] the score is negative.
    """
    # Get reranking agent
    reranker = SetwiseLlmRanker(
        model_name_or_path=model_path,
        tokenizer_name_or_path=model_path,
        device=kwargs.get('device', 'cuda'),
        num_child=kwargs.get('num_child', 10),
        k=top_k,
        scoring='likelihood',
        method='heapsort',
    )

    # Transform the retrieval runs into candidates
    pairs = convert_runs_to_pairs(topics=topics, corpus=corpus, runs=runs, top_k=top_k)

    # Rerank 
    rerank_results = []
    for i, qid in tqdm(enumerate(runs), total=len(runs)):
        rerank_result = reranker.rerank(*pairs[i])
        # ignore the doc field in the result 
        rerank_results.append(
            Result(
                query=Query(qid=qid, text=""),
                candidates=[Candidate(docid=r.docid, score=r.score, doc={"text": ""}) for r in rerank_result]
            )
        )

    # final postprocessing
    outputs = {}
    for rr in rerank_results:
        rr.candidates = rr.candidates[:top_k]

        query = rr.query
        candidates = rr.candidates
        outputs[query.qid] = {c.docid: c.score for c in candidates}

    # write
    if writer is not None:
        for qid in outputs: 
            hits = outputs[qid]
            sorted_result = {k: v for k,v in sorted(hits.items(), key=itemgetter(1), reverse=True)} 
            for i, (docid, score) in enumerate(sorted_result.items()):
                writer.write(f"{qid} Q0 {docid} {str(i+1)} {score} {model_path}\n")
        writer.close()

    return outputs
