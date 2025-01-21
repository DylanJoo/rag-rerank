""" 
[TODO] document segmentation and aggregation
[TODO] think about long-document scenario. Chunk-and-aggregate?
"""

from operator import itemgetter
import os
import datetime
import logging
import argparse 
from tqdm import tqdm
import json
import torch

from tools.ranking_utils import (
    load_runs, 
    load_corpus,
    load_topics, 
    batch_iterator
)
from augment.pointwise.reranking.utils import load_reranker

def filter(
    topics, corpus, runs,
    reranker_config,
    top_k, batch_size,
    max_length,
    threshold,
    writer=None
):

    reranker = load_reranker(**reranker_config)

    qids = list(topics.keys())
    qids = [qid for qid in qids if qid in runs]  # only appeared in run

    outputs = {}
    for qid in tqdm(qids, total=len(qids)):

        result = runs[qid]
        query = topics[qid]
        documents = [corpus[docid] for docid in result]

        # predict
        scores = []
        for batch_docs in batch_iterator(documents, batch_size):
            queries = [query] * len(batch_docs)
            batch_scores = reranker.predict(
                queries=queries,
                documents=[doc['text'] for doc in batch_docs],
                titles=[doc['title'] for doc in batch_docs],
                max_length=max_length
            )
            scores.extend(batch_scores)

        # sort candidates
        hits = {docid: scores[i] for i, docid in enumerate(result)}            
        sorted_result = {k: v for k,v in sorted(hits.items(), key=itemgetter(1), reverse=True)} 
        outputs[qid] = sorted_result

        # filter candidates
        if threshold is not None:
            outputs[qid] = {k: v for k,v in sorted_result.items() if v >= filter_threshold}
            outputs[qid] = sorted_result

        # write
        if writer is not None:
            for i, (docid, score) in enumerate(sorted_result.items()):
                writer.write(f"{qid} Q0 {docid} {str(i+1)} {score} {reranker}\n")

    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic_file", type=str, default=None)
    parser.add_argument("--corpus_dir_or_file", type=str, default=None)
    parser.add_argument("--run_file", type=str, default=None)

    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("-bs", "--batch_size_per_query", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=-1)

    # reranker config
    parser.add_argument("--reranker_class", type=str, default=None)
    parser.add_argument("--reranker_name_or_path", type=str, default=None)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--fp16", default=False, action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)
    writer = open(args.output, 'w')

    ## load data
    topics = load_topics(args.topic_file)
    corpus = load_corpus(args.corpus_dir_or_file)
    runs = load_runs(args.run_file, topk=args.top_k, output_score=True)

    with torch.no_grad():
        rerank(
            topics=topics, corpus=corpus, input_run=runs,
            reranker_config=vars(args),
            top_k=args.top_k, batch_size=args.batch_size_per_query,
            max_length=args.max_length,
            writer=writer
        )
    writer.close()

    print('done')
