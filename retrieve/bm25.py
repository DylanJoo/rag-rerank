import os
import json
import argparse
from collections import defaultdict 
from tqdm import tqdm 
from retrieve.utils import load_topic, batch_iterator
from pyserini.search.lucene import LuceneSearcher

def search(index, k1, b, topics, batch_size, k, writer=None):

    searcher = LuceneSearcher(index)
    searcher.set_bm25(k1=k1, b=b)

    qids = list(topics.keys())
    qtexts = list(topics.values())

    outputs = {}

    for (start, end) in tqdm(
            batch_iterator(range(0, len(qids)), batch_size, True),
            total=(len(qids)//batch_size)+1
    ):
        qids_batch = qids[start: end]
        qtexts_batch = qtexts[start: end]
        hits = searcher.batch_search(
                queries=qtexts_batch, 
                qids=qids_batch, 
                threads=32,
                k=k,
        )

        for key, value in hits.items():
            outputs[key] = {h.docid: h.score for h in hits[key]}

            for i in range(len(hits[key])):
                if writer is not None:
                    writer.write(
                        f'{key} Q0 {hits[key][i].docid} {i+1} {hits[key][i].score:.5f} bm25\n'
                    )
    return outputs  # could be a writer or a dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k1",type=float, default=0.9) # 0.5 # 0.82
    parser.add_argument("--b", type=float, default=0.4) # 0.3 # 0.68
    parser.add_argument("--k", default=100, type=int)
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--topics", default=None, type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--output", default=None, type=str)
    args = parser.parse_args()

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)
    writer = open(args.output, 'w')

    ## load data
    topics = load_topic(args.topics)

    search(
        index=args.index, 
        k1=args.k1, 
        b=args.b, 
        topics=topics,
        batch_size=args.batch_size, 
        k=args.k,
        output=output
    )
    writer.close()

    print('done')
