import os
import glob
import collections 
import json
from tqdm import tqdm

# def load_searcher(path, dense=False):
#     if dense:
#         from pyserini.search.faiss import FaissSearcher
#         searcher = FaissSearcher(path, None)
#     else:
#         from pyserini.search.lucene import LuceneSearcher
#         searcher = LuceneSearcher(path)
#         searcher.set_bm25(k1=0.9, b=0.4)
#     return searcher

def load_topic(path, use_answer=False):
    topic = {}
    with open(path) as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            qid = data['qid']
            qtext = data['question_text']
            topic[qid.strip()] = qtext.strip()

    return topic

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def load_topic(path):
    topic = {}
    with open(path, 'r') as f:
        for line in f:
            qid, qtext = line.split('\t')
            topic[str(qid.strip())] = qtext.strip()
    return topic

def load_corpus(path):
    # empty = {'title': "", 'text': ""}
    # corpus = collections.defaultdict(lambda: empty)
    corpus = {}
    id_field = 'id'
    content_field = 'contents'

    if os.path.isdir(path):
        files = [f for f in glob.glob(f'{path}/*')]
    else:
        files = [path]

    for file in files:
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                docid = data[id_field] # two datasets have diff identifier
                title = data.get('title', "").strip()
                text = data.get(content_field, "").strip()
                corpus[str(docid)] = {'title': title, 'text': text}
    return corpus

def load_runs(path, topk=10, output_score=False): # support .trec file only
    run_dict = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if int(rank) <= topk:
                run_dict[str(qid)] += [(docid, float(rank), float(score))]

    # sort by score and return static dictionary
    sorted_run_dict = collections.OrderedDict()
    for qid, docid_ranks in run_dict.items():
        sorted_docid_ranks = sorted(docid_ranks, key=lambda x: x[1], reverse=False) 
        if output_score:
            sorted_run_dict[qid] = [(docid, rel_score) for docid, rel_rank, rel_score in sorted_docid_ranks]
        else:
            sorted_run_dict[qid] = [docid for docid, _, _ in sorted_docid_ranks]

    return sorted_run_dict
