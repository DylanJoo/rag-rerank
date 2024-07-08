import os
import re
import argparse
import json
from tqdm import tqdm
from glob import glob

def remove_citations(sent):
    sent = sent.strip()
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def load_claims(path):
    data = json.load(open(path, 'r'))

    nuggets = []
    claims = []
    for i, item in enumerate(data['data']):
        example_id = f"mds-claims_{data['args']['shard']}-{i}" 
        if len(item['output']) == 0:
            continue
        outputs = item['output'][0].strip().split('\n')[:10]
        ndocs = item['number_of_documents']
        nuggets.append({
            "id": example_id, 
            "contents": [remove_citations(n).strip() for n in outputs],
            "type": 'nugget',
            "ndocs": ndocs
        })

        for j in range(1, ndocs+1): # the first one is nuggets
            example_doc_id = f"mds-claims_{data['args']['shard']}-{i}#{j}" 
            outputs = item['output'][j].strip().split('\n')[:10]
            claims.append({
                "id": example_doc_id, 
                "contents": [remove_citations(n).strip() for n in outputs],
                "type": 'claim',
            })
    return nuggets, claims

def load_question(path):
    data = json.load(open(path, 'r'))

    questions = []
    for i, item in enumerate(data['data']):
        example_id = f"mds-question_{data['args']['shard']}-{i}" 
        outputs = item['output'].strip().split('?')[0] + "?"
        questions.append({
            "id": example_id, 
            "contents": outputs,
            "type": 'question',
        })

    return questions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("-input", "--input_dir", type=str, default=None)
    parser.add_argument("-output", "--output_dir", type=str, default=None)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True) 

    # questions
    questions_all = []
    files_questions = glob(os.path.join(args.input_dir, "*question*"))
    for file in tqdm(files_questions):
        question = load_question(file) # one per example
        questions_all += question

    with open(os.path.join(args.output_dir, "questions.jsonl"), 'w') as f:
        for question in questions_all:
            f.write(json.dumps(question) +'\n')

    # claims and nuggets
    claims_all = []
    nuggets_all = []
    files_claims = glob(os.path.join(args.input_dir, "*claims*"))
    for file in tqdm(files_claims):
        nuggets, claims = load_claims(file)
        nuggets_all += nuggets
        claims_all += claims

    with open(os.path.join(args.output_dir, "claims.jsonl"), 'w') as f:
        for claim in claims_all:
            f.write(json.dumps(claim) +'\n')

    with open(os.path.join(args.output_dir, "nuggets.jsonl"), 'w') as f:
        for nugget in nuggets_all:
            f.write(json.dumps(nugget) +'\n')
