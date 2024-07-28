import os
import re
import argparse
import json
from tqdm import tqdm
from glob import glob
from collections import defaultdict

def remove_citations(sent):
    sent = sent.strip()
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def replace_citations(sent):
    sent = re.sub(r"\[\d+\]", "\n", sent)
    pattern = re.compile(r"\n+")
    sent = re.sub(pattern, '\n', sent)
    return sent

def normalize_texts(texts, size=10000):
    texts = remove_citations(texts)
    texts = texts.strip()
    pattern = re.compile(r"\n")
    texts = re.sub(pattern, ' ', texts).strip()
    pattern = re.compile(r"\s+")
    texts = re.sub(pattern, ' ', texts).strip()
    texts = maybe_truncation(texts, size)
    return texts

def maybe_truncation(text, size=10000):
    if len(text) > size:
        words = text.split(' ')[:(size//10)]
        return " ".join(words)
    else:
        return text

def load_nuggets_and_claims(path, n=10):
    data = json.load(open(path, 'r'))

    nuggets = []
    claims = []
    for i, item in enumerate(data['data']):
        example_id = item["example_id"]
        if len(item['output']) == 0:
            continue
        outputs = replace_citations(item['output'][0]).split('\n')[:n]
        ndocs = item['number_of_documents']
        nuggets.append({
            "example_id": example_id, 
            "contents": [normalize_texts(n).strip() for n in outputs],
            "full_text": normalize_texts(item['full_text'][0]), # we dont need this
            "type": 'nuggets',
            "ndocs": ndocs
        })

        for j in range(1, ndocs+1): # the first one is nuggets
            example_doc_id = f"{example_id}#{j}" 
            outputs = item['output'][j].strip().split('\n')[:n]
            claims.append({
                "example_doc_id": example_doc_id, 
                "contents": [normalize_texts(n).strip() for n in outputs],
                "full_text": normalize_texts(item['full_text'][j]),
                "type": 'claims',
            })
    return nuggets, claims

def load_question(path):
    data = json.load(open(path, 'r'))

    questions = []
    for i, item in enumerate(data['data']):
        # example_id = item['example_id'].replace('question', 'claims')
        # outputs = item['output'].strip().split('?')[0] + "?"
        example_id = item['example_id']
        example_id = item['example_id'].replace('question', 'claims')
        outputs = item['output'].strip().split('</query>')[0]
        outputs = outputs.replace("?", "")
        questions.append({
            "id": example_id,
            "contents": outputs,
            "type": 'question',
        })
    return questions

def load_result_to_dict(path, n=10, threshold=-99):
    results = defaultdict(list)

    with open(path, 'r') as f:
        for line in f:
            query_id, example_doc_claim_id, rank, score = line.split()
            example_id = query_id.rsplit(':', 1)[0]
            rank = int(rank)
            score = float(score)
            # type_of_premise = "nugget" if ":" in query_id else "fulltext"

            if len(results[query_id]) >= n:
                continue

            # for the concentrator. highere score and in the same example
            if (score >= threshold) and (example_id in example_doc_claim_id): 
                results[query_id].append({
                    "id": example_doc_claim_id, "rank": rank, "score": score
                })

            # for the concentrator. lower score but not in the example
            if (score < threshold) and (example_id not in example_doc_claim_id): 
                results[query_id].append({
                    "id": example_doc_claim_id, "rank": rank, "score": score
                })

    return results
