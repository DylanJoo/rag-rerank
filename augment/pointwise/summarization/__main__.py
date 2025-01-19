""" 
[TODO] Do not implment the non query-focused versioDo not implment the non query-focused version
[TODO] think about adding the feature that can load pre-summarized results 
[TODO] Still summarize also the irrelevant documents, only skip when evaluation.
[TODO] think about long-document scenario. Chunk-and-summarize? 
"""
import os
import torch
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

from tools.ranking_utils import (
    load_runs, 
    load_corpus, 
    load_topics,
    batch_iterator, 
)
from augment.pointwise.summarization.utils import load_summarizer

def summarize(
    topics, corpus, runs,
    summarizer_config,
    top_k, batch_size,
    max_length,
    template="{d}",
    writer=None,
):

    summarizer = load_summarizer(**summarizer_config)

    qids = list(topics.keys())

    outputs = {}

    for qid in tqdm(qids, total=len(qids)):

        result = runs[qid]
        topic = topics[qid]
        documents = [corpus[docid] for i, docid in enumerate(result)][:top_k]

        # predict
        summaries = []
        for batch_docs in batch_iterator(documents, batch_size):
            inputs = list(
                template.replace('{q}', topic).replace('{d}', (doc['title']+doc['text']).strip() ) \
                        for doc in batch_docs
            )
            batch_outputs = summarizer.generate(inputs, min_tokens=32, max_tokens=512)
            summaries.extend(batch_outputs)

        outputs[qid] = summaries

        # write
        if writer is not None:
            writer.write(json.dumps({
                "qid": qid, "topic": topic, "contexts": summaries, 
                "context_type": f"{summarizer_config['summarizer_name_or_path']}_{top_k}",
                "docids": [docid for docid in result], 
            }, ensure_ascii=False)+'\n')

    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic_file", type=str, default=None)
    parser.add_argument("--corpus_dir_or_file", type=str, default=None)
    parser.add_argument("--run_file", type=str, default=None)

    parser.add_argument("--template", type=str, default="{d}")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument('-bs', "--batch_size_per_query", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--summarizer_class", type=str, default=None)
    parser.add_argument("--summarizer_name_or_path", type=int, default=None)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--fp16", default=False, action='store_ture')
    args = parser.parse_args()

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)
    writer = open(args.output, 'w')

    ## load data
    topics = load_topic(args.topic_file)
    corpus = load_corpus(args.corpus_dir_or_file)
    runs = load_runs(args.run_file, topk=args.top_k, output_score=True)

    with torch.no_grad():
        summarize(
            topics=topics, corpus=corpus, input_run=runs,
            summarizer_config=vars(args),
            top_k=args.top_k,
            batch_size=args.batch_size_per_query,
            max_length=args.max_length,
            template=args.template,
            writer=writer
        )
    writer.close()

    print('done')

