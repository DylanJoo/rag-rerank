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

    # model, tokenizer = load_model(args.model_name_or_path, model_class='fid')
    # model.eval()
    #
    # eval_data = load_dataset('json', data_files=args.eval_file, keep_in_memory=True)['train']
    # collator = Standard
    #
    # for eval_data_item in tqdm(eval_data, total=len(eval_data)):
    #
    #     request = eval_data_item['question']
    #
    #     summaries = []
    #     for batch_docs in batch_iterator(eval_data_item['doc_ctxs'], args.n_contexts):
    #
    #         ## multi-doc summarization
    #         if 'prefix' in args.model_name_or_path:
    #             tokenized_inputs = collator
    #             tokenized_input = tokenizer(input, padding=True, truncation=True, max_length=512, return_tensors='pt').to(model.device)
    #
    #             outputs = generate_standard_with_prefix()
    #         else:
    #             outputs = generate_standard()
    #
    #
    #         summaries.extend(outputs)
    #
    #     # add the new summaries 
    #     for i, summary in enumerate(summaries):
    #         eval_data_item['docs'][i][args.output_key] = summary
    #
    # if not os.path.exists("data/add_summary"):
    #     os.makedirs("data/add_summary")
    # json.dump(eval_data, open(f"data/add_summary/{args.output_file}", "w"), indent=4)
