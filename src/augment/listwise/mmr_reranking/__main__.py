"""
As bi-encoder is not the focus here, we adopt the sentence transformer pre-trained embedder for MMR.
[TODO] See if we need to build model class like pointwise reranking.
"""
from operator import itemgetter
import os
import torch
import json
import argparse
from tqdm import tqdm

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def mmr_process(
    qembed, dembeds, 
    k, lambda_param=0.5
):

    sim_scores = cosine_similarity(qembed.reshape(1, -1), dembeds).flatten()
    dd_sim_scores = cosine_similarity(dembeds)

    selected_docs = []
    selected_docs.append(np.argmax(sim_scores))

    remaining_docs = list(range(len(dembeds)))
    remaining_docs.remove(selected_docs[0])

    while len(selected_docs) < k:
        max_score = -np.inf
        next_doc = None

        for i in remaining_docs:
            sim = sim_scores[i]
            div = np.max(dd_sim_scores[i, selected_docs]) if selected_docs else 0
            mmr_score = lambda_param * sim - (1 - lambda_param) * div

            if mmr_score > max_score:
                max_score = mmr_score
                next_doc = i

        selected_docs.append(next_doc)
        remaining_docs.remove(next_doc)

    return selected_docs + remaining_docs # use the last ranking process as remaining's

def mmr_rerank(
    topics, corpus, runs,
    encoder_name_or_path,
    max_k, batch_size,
    lambda_param=0.5,
    max_length=512,
    writer=None
):
    encoder = SentenceTransformer(encoder_name_or_path)

    qids = list(topics.keys())
    qids = [qid for qid in qids if qid in runs]  # only appeared in run
    queries = [topics[qid] for qid in qids]

    # get query embeddings
    qembeds = encoder.encode(queries, batch_size=batch_size)

    # get document embeddings for each query and do mmr
    outputs = {}
    for i, qid in enumerate(tqdm(qids, total=len(qids))):

        result = runs[qid]

        # get document embdedings
        documents = [corpus[docid] for docid in result]
        max_k = min(max_k, len(documents))
        dembeds = encoder.encode([ (doc['title'] + " " + doc['text']).strip() for doc in documents], batch_size=batch_size)

        # mmr 
        mmr_orders = mmr_process(
            qembed=qembeds[i], 
            dembeds=dembeds, 
            k=max_k, lambda_param=lambda_param
        )

        # sort candidates
        docids = [docid for docid in result]
        hits = {docids[order]: 1/(i+1) for i, order in enumerate(mmr_orders)}            
        sorted_result = {k: v for k,v in sorted(hits.items(), key=itemgetter(1), reverse=True)} 
        outputs[qid] = sorted_result

        # write
        if writer is not None:
            for i, (docid, score) in enumerate(sorted_result.items()):
                writer.write(f"{qid} Q0 {docid} {str(i+1)} {score} {encoder_name_or_path}-MMR:{lambda_param}\n")
            writer.close()

    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_key", type=str, default="summary_debug")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--template", type=str, default="title: {T} content: {P}")

    args = parser.parse_args()
