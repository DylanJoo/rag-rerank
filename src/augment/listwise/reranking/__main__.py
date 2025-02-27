"""
The listwise ranking is mostly relying on `rank_llm` repositary.
See details in `https://github.com/castorini/rank_llm`
"""
import copy
from typing import Any, Dict, List, Union
from operator import itemgetter

from rank_llm.data import Query, Request, Candidate
from rank_llm.rerank import IdentityReranker, RankLLM, Reranker, PromptMode
from rank_llm.rerank.listwise import RankListwiseOSLLM


def convert_runs_to_requests(topics=None, corpus=None, runs=None, top_k=50):
    requests = []
    for qid in runs:

        candidates = []
        top_k_runs = runs[qid]
        for i, docid in enumerate(runs[qid]):
            if i < top_k:
                candidates.append(
                    Candidate(docid=docid, score=runs[qid][docid], doc=corpus[docid])
                )

        requests.append(
                Request(
                    query=Query(text=topics[qid], qid=qid),
                    candidates=candidates
                )
        )
    return requests

def rerank2(
    topics, corpus, runs, 
    model_path: str,
    top_k: int = 100,
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    num_passes: int = 1,
    interactive: bool = False,
    default_model_coordinator: RankLLM = None,
    writer=None,
    **kwargs
):

    # Adopt arguemnts to RankLLM
    kwargs['prompt_mode'] = PromptMode.RANK_GPT if kwargs['prompt_mode'] == 'rank_GPT' else None
    kwargs['context_size'] = kwargs.get('context_size', 4096)

    # Transform the retrieval runs into requests
    requests = convert_runs_to_requests(
        topics=topics, corpus=corpus, runs=runs, top_k=top_k, 
    )
    from rank_llm.rerank.listwise import ZephyrReranker
    reranker = ZephyrReranker()
    rerank_results = reranker.rerank_batch(requests=requests)

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

def rerank(
    topics, corpus, runs, 
    model_path: str,
    top_k: int = 100,
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    num_passes: int = 1,
    interactive: bool = False,
    default_model_coordinator: RankLLM = None,
    writer=None,
    **kwargs
):

    # Adopt arguemnts to RankLLM
    kwargs['prompt_mode'] = PromptMode.RANK_GPT if kwargs['prompt_mode'] == 'rank_GPT' else None
    kwargs['context_size'] = kwargs.get('context_size', 4096)

    # Get reranking agent
    reranker = Reranker(
        Reranker.create_model_coordinator(model_path, default_model_coordinator, interactive, **kwargs)
    )


    # Transform the retrieval runs into requests
    requests = convert_runs_to_requests(
        topics=topics, corpus=corpus, runs=runs, top_k=top_k, 
    )

    # Reranking stages
    print(f"Reranking and returning {top_k} passages with {model_path}...")
    for pass_ct in range(num_passes):
        print(f"Pass {pass_ct + 1} of {num_passes}:")

        rerank_results = reranker.rerank_batch(
            requests,
            rank_end=top_k,
            rank_start=0,
            shuffle_candidates=shuffle_candidates,
            logging=print_prompts_responses,
            top_k_retrieve=top_k,
            vllm_batched=kwargs.pop('vllm_batched', True),
            batch_size=kwargs.pop('batch_size', 32),
            **kwargs,
        )

    if num_passes > 1:
        requests = [
            Request(copy.deepcopy(r.query), copy.deepcopy(r.candidates))
            for r in rerank_results
        ]

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
    # if interactive:
    #     return (rerank_results, reranker.get_agent())
    # else:
    #     return rerank_results
