import json
import random
import numpy as np

# randomly selected among 50
random.seed(327)
random_qids = [f'duc04-testb-{i}' for i in random.sample(range(50), 10)]

from tools import load_judgements, load_qrels

def export_query_file(file, writer, threshold=3):

    judgements = load_judgements(f"/home/dju/datasets/crux/ranking_{threshold}/testb_oracle-passages_judgements.jsonl")
    qrels = load_qrels(f"/home/dju/datasets/crux/ranking_{threshold}/testb_qrels_pr.txt")
    rel_threshold = 3

    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            qid = data['qid']
            if qid in random_qids:
                data['k'] = data['type'][1]
                data['type'] = data['type'][0]

                # check only the answerable
                docids = [docid for docid, score in qrels[qid].items() if score >= rel_threshold ] 
                judgement_oracle = np.array([judgements[qid][docid] for docid in docids]).max(0)
                answerable = (judgement_oracle >= threshold) 

                data['questions'] = [q if ans_flag else None for q, ans_flag in zip(data['questions'], answerable)]

                writer.write(json.dumps(data) + "\n")
    writer.close()

def export_run_file(file, writer, prefix=''):
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            qid = data['qid']
            if qid in random_qids:
                for docid in data['docids']:
                    writer.write(f"{qid} Q0 {docid} {i+1} {1/(i+1)} human_eval{prefix}\n")
    writer.close()

def export_passages(file):
    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            docid = data['id']
            contents = data['contents']
            with open(f"results/human_eval/passages/{docid}", 'w') as w:
                w.write(json.dumps(data) + '\n')

## 0. Get passage
# corpus_file = '/home/dju/datasets/crux/passages/testb_psgs.jsonl'
# export_passages(corpus_file)

## 1. oracle report
result_file = 'results/oracle/testb-oracle_k-1:3_n-1.jsonl'
writer = open('results/human_eval/oracle.jsonl', 'w')
export_query_file(result_file, writer)

writer = open('results/human_eval/oracle.run', 'w')
export_run_file(result_file, writer, prefix='_oracle')

## 2. bm25 report
result_file = 'results/-1/testb-bm25_100-vanilla_-1.jsonl'
writer = open('results/human_eval/bm25.jsonl', 'w')
export_query_file(result_file, writer)

writer = open('results/human_eval/bm25.run', 'w')
export_run_file(result_file, writer, prefix='_bm25')

## 3. contriever+rankfirst report
result_file = 'results/-1/testb-bm25_100-rankfirst_100-vanilla_-1.jsonl'
writer = open('results/human_eval/dr_rankfirst.jsonl', 'w')
export_query_file(result_file, writer)

writer = open('results/human_eval/dr_rankfirst.run', 'w')
export_run_file(result_file, writer, prefix='_dr_rankfirst')
