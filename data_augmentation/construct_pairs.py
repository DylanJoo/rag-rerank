import os
import re
import argparse
import json
import random
from collections import defaultdict
from tqdm import tqdm
from glob import glob
from utils import (
    normalize_texts, 
    maybe_truncation,
    load_question,
    load_nuggets_and_claims,
    load_result_to_dict
)

def get_claims(example_id, doc_id, claim_idx):
    id = f"{example_id}#{doc_id}:{claim_idx}"

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--result_jsonl", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--n_max_claims", type=int, default=None)
    parser.add_argument("--n_max_docs", type=int, default=None)

    # data augmentation
    args = parser.parse_args()
    os.makedirs(args.output_jsonl.rsplit('/', 1)[0], exist_ok=True) 

    ## questions
    questions_all = []
    files = glob(os.path.join(args.input_dir, "*question*.json"))
    for file in tqdm(files):
        question = load_question(file) # one per example
        questions_all += question
    questions = {q['id']: q['contents'] for q in questions_all}

    ## fulltext or claims 
    claims_all = []
    nuggets_all = []
    files = glob(os.path.join(args.input_dir, "*claims*.json"))
    for file in tqdm(files):
        nuggets, claims = load_nuggets_and_claims(file)
        nuggets_all += nuggets
        claims_all += claims

    example_ndocs = {n['example_id']: n['ndocs'] for n in nuggets_all}
    doc_fulltexts = {}
    doc_claims = {}
    for claim in claims_all:
        example_doc_id = claim['example_doc_id']
        doc_fulltexts[example_doc_id] = claim['full_text']
        for i, content in enumerate(claim['contents']):
            example_doc_claim_id = f"{example_doc_id}:{i}"
            doc_claims[example_doc_claim_id] = content

    ## result
    f = open(args.output_jsonl, 'w')
    results = load_result_to_dict(
        args.result_jsonl, 
        n=args.n_max_claims,
        threshold=-1
    ) # top_k claims

    for query_id in results: 
        ndoc = example_ndocs[query_id] # query_id == example_id
        align_doc_claim_ids = [r['id'] for r in results[query_id]]

        ## get doc sources
        concentrators, distractors = [], []
        for id in align_doc_claim_ids:
            if query_id in id.rsplit(":", 1)[0]:
                concentrators.append(id)
            else:
                distractors.append(id)

        ## candidate document claims
        concentrate_pairs = defaultdict(list)
        disctract_pairs = defaultdict(list)

        for target_claim_id in concentrators:
            docid = target_claim_id.rsplit(':', 1)[0] # {example_id}#{doc_id}:{claim_id}
            concentrate_pairs[docid].append(target_claim_id)

        for target_claim_id in distractors:
            docid = target_claim_id.rsplit(':', 1)[0] # {example_id}#{doc_id}:{claim_id}
            disctract_pairs[docid].append(target_claim_id)

        ## compose into pairs
        labels = []
        full_ctx = []
        comp_ctx = []

        for example_doc_id, list_of_claim_ids in concentrate_pairs.items():
            full_ctx.append(
                normalize_texts(doc_fulltexts[example_doc_id], size=6400)
            )
            labels.append(1)

            comp_ctx_claim = []
            for example_doc_claim_id in list_of_claim_ids:
                comp_ctx_claim.append(doc_claims[example_doc_claim_id])
            comp_ctx.append(comp_ctx_claim)

        for example_doc_id, list_of_claim_ids in disctract_pairs.items():

            if len(labels) >= args.n_max_docs:
                continue

            full_ctx.append(
                normalize_texts(doc_fulltexts[example_doc_id], size=5120)
            )
            labels.append(0)

            comp_ctx_claim = []
            for example_doc_claim_id in list_of_claim_ids:
                comp_ctx_claim.append(doc_claims[example_doc_claim_id])
            comp_ctx.append(comp_ctx_claim)

        if sum(labels) > 0:
            f.write(json.dumps({
                "question": questions[query_id],
                "doc_ctxs": full_ctx,
                "comp_ctxs": comp_ctx,
                "labels": labels,
            }, indent=4)+'\n')

    f.close()


