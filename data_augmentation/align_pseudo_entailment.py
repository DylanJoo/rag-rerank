import torch
import os
import argparse
import logging
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from utils import load_nuggets_and_claims, load_question
from pyserini.search.lucene import LuceneSearcher
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None
AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"

def mine_distractor(
    query, 
    searcher, 
    k=100,
    max_docs_return=3,
    ignored_prefix="#"
):

    hits = searcher.search(query, k)
    hit_doc_ids = []

    # filter the ignored
    ### schema: {example_id}#{n_doc}:{n_claims}
    for h in hits:
        if h.docid.split("#")[0] != ignored_prefix:
            hit_doc_ids.append(h.docid.split(":")[0])

    hit_doc_ids = sorted(set(hit_doc_ids), key=hit_doc_ids.index)[:max_docs_return]
    return hit_doc_ids

def _run_nli_autoais(nugget, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer, label_ids
    input_text = "premise: {} hypothesis: {}".format(nugget, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(
            input_ids, max_new_tokens=10, 
            return_dict_in_generate=True, output_scores=True
        )

    result = autoais_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    result = outputs.scores[0][0, label_ids].softmax(-1) # first token and the b
    score = result[1].item()
    return inference, score

def compute_claims(nugget, claims):
    global autoais_model, autoais_tokenizer, label_ids
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
        label_ids = autoais_tokenizer.convert_tokens_to_ids(["0", "1"])

    logger.info("Computing claims...")
    scores = []
    predictions = []
    for item in claims:
        for claim in claims:
            entail, score = _run_nli_autoais(nugget, claim)
            predictions.append(entail)
            scores.append(score)

    return predictions, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--result_jsonl", type=str, default=None)
    parser.add_argument("--doc_claims_index", type=str, default=None)
    args = parser.parse_args()

    # questions
    # questions_all = []
    # files_questions = glob(os.path.join(args.input_dir, "*question*.json"))
    # for file in tqdm(files_questions):
    #     question = load_question(file) # one per example
    #     questions_all += question

    # claims and nuggets
    claims_all = []
    nuggets_all = []
    files_claims = glob(os.path.join(args.input_dir, "*claims*.json"))
    for file in tqdm(files_claims):
        nuggets, claims = load_nuggets_and_claims(file)
        nuggets_all += nuggets
        claims_all += claims
        print(nuggets[0]['contents'])

    # reverse the claims to mapping
    claims_all_dict = defaultdict(list)
    for claims in claims_all:
        example_doc_id = claims.pop('example_doc_id')
        print(example_doc_id)
        print(claims['contents'][0])
        exit(0)
        claims_all_dict[example_doc_id] = claims

    ## align claims to nugget
    for nuggets_dict in tqdm(nuggets_all[:2]):
        example_id = nuggets_dict['example_id']
        nuggets = nuggets_dict['contents']
        ndocs = nuggets_dict['ndocs']

        claim_ids = []
        claim_texts = []
        claim_scores = []
        claim_predictions = []
        ### the potentially entailed claims from ndocs
        for n in range(1, 1+ndocs):
            example_doc_id = f"{example_id}#{n}"
            claims = claims_all_dict[example_doc_id]
            claim_ids += [f"{example_doc_id}:{i}" for i in range(len(claims))]
            claim_texts += claims['contents']

        print(example_id)
        print(nuggets)
        print(claim_texts)
        exit(0)

        for nugget in nuggets:

            ### the potentially contradicted claims
            if args.doc_claims_index is not None:
                searcher = LuceneSearcher(args.doc_claims_index)
                dis_example_doc_ids = mine_distractor(
                    query=nugget, 
                    searcher=searcher, 
                    k=100,
                    max_docs_return=1,
                    ignored_prefix=example_id
                )

                for example_doc_id in dis_example_doc_ids:
                    claims = claims_all_dict[example_doc_id]
                    claim_ids += [f"{example_doc_id}:{i}" for i in range(len(claims))]
                    claim_texts += claims['contents']

            ### predict the pseudo entilament for each nugget
            predictions, scores = compute_claims(nugget, claim_texts)
            claim_scores += scores
            claim_predictions += predictions

            logger.info(scores)
            logger.info(predictions)

