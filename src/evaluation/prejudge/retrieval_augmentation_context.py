import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import re
import argparse
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import ir_measures
from ir_measures import RPrec, R, MAP, nDCG, alpha_nDCG
from transformers import AutoTokenizer
from tools import load_judgements, sort_and_truncate

def rac_evaluate(
    corpus, qrels, judgements, diversity_qrels,
    rac_data,
    n_questions,
    threshold=0,     # answerability threshold (tau)
    rel_threshold=3, # on qrel's last column
    runs=None,
    tokenizer_name='meta-llama/Llama-3.1-70B-Instruct',
    gamma=0.5, tag='experiment'
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    outputs = {'coverage': [], 'density': [], 'num_segs': [], 'num_tokens': []}

    overlapped = {k: v for k, v in qrels.items() if k in rac_data}
    max_k = {}

    if len(overlapped) != len(qrels):
        logger.warning(' #Topics in qrels and rac_data are not consistent.' + \
                f' Got {len(qrels)} and {len(rac_data)}.')
        qrels = overlapped

    for qid in tqdm(qrels, desc='RAC Evaluating', total=len(qrels)):

        # [oracle] 
        docids = [docid for docid, score in qrels[qid].items() if score >= rel_threshold ] 
        rac_text = " ".join([corpus[docid]['text'] for docid in docids])
        n_tokens = len(tokenizer.tokenize(rac_text))

        judgement_oracle = np.array([judgements[qid][docid] for docid in docids]).max(0)
        answerable = (judgement_oracle >= threshold)
        density_oracle = sum(answerable) / n_tokens

        # [retrieval-augmented context] 
        rac_type = rac_data[qid]['type']
        rac_text = " ".join(rac_data[qid]['context_list'])
        docids = rac_data[qid]['docids']
        max_k[qid] = rac_type[1] # align to the max_k used in rac

        if 'oracle-report' in tag:
            docids = [f'{qid}:report']

        ratings = [[0] * n_questions]
        for docid in docids:

            ## answerability 
            if qid == docid.split(":")[0]: # only consider the context derieved from relevant
                judgement = judgements[qid][docid]
                ratings.append(judgement)

        # print(ratings)
        ratings = np.array(ratings).max(0)

        # [calculate] coverage
        coverage = sum(ratings[answerable] >= threshold) / sum(answerable)

        # [calculate] density
        n_tokens = len(tokenizer.tokenize(rac_text)) 
        density = sum(ratings[answerable] >= threshold) / n_tokens
        norm_density = (density / density_oracle) ** gamma

        outputs['coverage'].append(coverage)
        outputs['density'].append(norm_density)
        outputs['num_segs'].append(len(docids))
        outputs['num_tokens'].append(n_tokens)

    # results
    mean_coverage = np.mean(outputs['coverage'])
    mean_density = np.mean(outputs['density'])
    mean_num_segments = np.mean(outputs['num_segs'])
    mean_num_tokens = np.mean(outputs['num_tokens'])
    num_coverage = len(outputs['coverage'])

    output_eval = {
        'mean_coverage': mean_coverage,
        'mean_density': mean_density,
        'mean_num_segments': mean_num_segments,
        'mean_num_tokens': mean_num_tokens,
        'num_coverage': num_coverage,
    }

    # results from ir_measures if have runs
    if runs is not None:
        runs = sort_and_truncate(runs, max_k) 
        rank_results = ir_measures.calc_aggregate([R@100, MAP, nDCG], qrels, runs)
        output_eval['Recall'] = rank_results[R@100] 
        output_eval['MAP'] = rank_results[MAP]
        output_eval['nDCG'] = rank_results[nDCG]

        rank_results = ir_measures.calc_aggregate([alpha_nDCG@20], diversity_qrels, runs)
        output_eval['alpha_nDCG'] = rank_results[alpha_nDCG@20]

    return output_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--index_dir", type=str, default='data/index')
    args = parser.parse_args()

    rac_evaluate(
        corpus, qrels, judgements, diversity_qrels,
        rac_data,
        n_questions,
        threshold=args.threshold,
        rel_threshold=args.rel_subset,
        runs=runs,
        tokenizer_name='bert-base-uncased',
        gamma=0.5, tag='experiment'
    )
