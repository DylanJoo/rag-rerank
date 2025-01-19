import os
import json
import argparse
from tqdm import tqdm

from tools.ranking_utils import (
    load_runs, 
    load_corpus, 
    load_topics,
)

def vanilla(
    topics, corpus, runs,
    top_k
    writer=None,
):

    qids = list(topics.keys())

    outputs = {}

    for qid in tqdm(qids, total=len(qids)):

        result = runs[qid]
        topic = topics[qid]
        documents = [corpus[docid] for i, docid in enumerate(result)][:top_k]
        raw_documents = [(d['title'] + " " + d['text']) for d in documents]

        if writer is not None:
            writer.write(json.dumps({
                "qid": qid, "topic": topic, "contexts": raw_documents, 
                "context_type": f"vanilla}_{top_k}",
                "docids": [docid for docid in result], 
            }, ensure_ascii=False)+'\n')

    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic_file", type=str, default=None)
    parser.add_argument("--corpus_dir_or_file", type=str, default=None)
    parser.add_argument("--run_file", type=str, default=None)

    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)
    writer = open(args.output, 'w')

    ## load data
    topics = load_topic(args.topic_file)
    corpus = load_corpus(args.corpus_dir_or_file)
    runs = load_runs(args.run_file, topk=args.top_k, output_score=True)

    vanilla(
        topics=topics, corpus=corpus, input_run=runs,
        top_k=args.top_k,
        writer=writer
    )
    writer.close()

    print('done')

