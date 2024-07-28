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

def deduplicate_and_sort(doc_ids):
    doc_ids = list(set(doc_ids))
    sorted(doc_ids)
    return doc_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--result_jsonl", type=str, default=None)
    parser.add_argument("--dataset_jsonl", type=str, default=None)
    parser.add_argument("--n_max_docs", type=int, default=None)
    parser.add_argument("--min_scores", type=float, default=-1)
    parser.add_argument("--n_claims_per_query", type=int, default=None)

    parser.add_argument("--good_read", action='store_true', default=False)

    args = parser.parse_args()
    os.makedirs(args.dataset_jsonl.rsplit('/', 1)[0], exist_ok=True) 

    # questions
    questions_all = []
    # files = glob(os.path.join(args.input_dir, "*question*.json"))
    files = glob(os.path.join(args.input_dir, "*topic*.json"))
    for file in tqdm(files):
        question = load_question(file) # one per example
        questions_all += question
    questions = {q['id']: q['contents'] for q in questions_all}

    # claims 
    claims_all, nuggets_all = [], []
    files = glob(os.path.join(args.input_dir, "*claims*.json"))
    for file in tqdm(files):
        nuggets, claims = load_nuggets_and_claims(file)
        nuggets_all += nuggets
        claims_all += claims

    claims = {}
    for c in claims_all:
        example_doc_id = c['example_doc_id']
        claims[example_doc_id] = c['full_text']
        for i, content in enumerate(c['contents']):
            example_doc_claim_id = f"{example_doc_id}:{i}"
            claims[example_doc_claim_id] = content

    # result
    f = open(args.dataset_jsonl, 'w')
    results = load_result_to_dict(
        args.result_jsonl, 
        n=args.n_claims_per_query,
        threshold=args.min_scores
    ) # top_k claims

    concentrate_pairs = defaultdict(list)
    distract_pairs = defaultdict(list)
    example_mapping = defaultdict(list)

    ### collect doc_claimss of each example
    for query_id in results: 

        ## [NOTE] we can calculate the length concentrators, maybe random one of top one.
        example_id = query_id.rsplit('#', 1)[0]
        align_doc_claim_ids = [r['id'] for r in results[query_id]]

        # concentrators, distractors = [], []
        for doc_claim_id in align_doc_claim_ids:
            doc_id = doc_claim_id.rsplit(':', 1)[0]

            ## positively aligned
            if example_id in doc_claim_id:
                example_mapping[example_id].append(doc_id)
                concentrate_pairs[doc_id].append(doc_claim_id)

            ## negatively aligned
            else:
                example_mapping[example_id].append(doc_id)
                distract_pairs[doc_id].append(doc_claim_id)

    ## compose into pairs
    for example_id, example_doc_ids in example_mapping.items():

        labels, full_ctx, comp_ctx, doc_ids = [], [], [], []
        for example_doc_id in set(example_doc_ids):
            fulltext = normalize_texts(claims[example_doc_id], size=6400)
            full_ctx.append(fulltext)

            label = int(example_id in example_doc_id)
            labels.append(label)

            if label == 1:
                comp_ctx_claim = []
                for id in deduplicate_and_sort(concentrate_pairs[example_doc_id]):
                    claim = claims[id]
                    comp_ctx_claim.append(claim)
                if len(comp_ctx_claim) > 0:
                    doc_ids.append(example_doc_id)
                    comp_ctx.append(comp_ctx_claim)

            if label == 0:
                comp_ctx_claim = []
                for id in deduplicate_and_sort(distract_pairs[example_doc_id]):
                    claim = claims[id]
                    comp_ctx_claim.append(claim)
                if len(comp_ctx_claim) > 0:
                    doc_ids.append(example_doc_id)
                    comp_ctx.append(comp_ctx_claim)

        if sum(labels[:args.n_max_docs]) > 0:
            f.write(json.dumps({
                "example_id": example_id,
                "doc_ids": doc_ids[:args.n_max_docs],
                "doc_ctxs": full_ctx[:args.n_max_docs],
                "question": questions[example_id],
                "comp_ctxs": comp_ctx[:args.n_max_docs],
                "labels": labels[:args.n_max_docs],
            }, indent=4 if args.good_read else None)+'\n')

    f.close()


