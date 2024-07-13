import os
import re
import argparse
import json
from tqdm import tqdm
from glob import glob
from utils import load_question, load_nuggets_and_claims

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("-input", "--input_dir", type=str, default=None)
    parser.add_argument("-output", "--output_dir", type=str, default=None)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True) 

    # claims and nuggets
    claims_all = []
    nuggets_all = []
    files_claims = glob(os.path.join(args.input_dir, "*claims*.json"))
    for file in tqdm(files_claims):
        nuggets, claims = load_nuggets_and_claims(file)
        nuggets_all += nuggets
        claims_all += claims

    with open(os.path.join(args.output_dir, "doc_claims.jsonl"), 'w') as f1:
         for claim in tqdm(claims_all, total=len(claims_all)):
            for idx, content in enumerate(claim['contents']):
                f1.write(json.dumps({
                    "id": f"{claim['example_doc_id']}:{idx}", # 
                    "contents": content
                })+'\n')

    # [NOTE] so far we dont need to flatten full text (as we dont index it)
    # with open(os.path.join(args.output_dir, "doc_claims.jsonl"), 'w') as f1, \
    #      open(os.path.join(args.output_dir, "doc_full_text.jsonl"), 'w') as f2:
    #         f2.write(json.dumps({
    #             "id": {claim['example_doc_id']},
    #             "content": claim['full_text']
    #         })+'\n')
    # [NOTE] so far we dont need to flatten nuggets
    with open(os.path.join(args.output_dir, "nuggets.jsonl"), 'w') as f:
        for nugget in nuggets_all:
            f.write(json.dumps(nugget) +'\n')

    # questions
    # [NOTE] so far we dont need to flatten question
    # questions_all = []
    # files_questions = glob(os.path.join(args.input_dir, "*question*"))
    # for file in tqdm(files_questions):
    #     question = load_question(file) # one per example
    #     questions_all += question
    #
    # with open(os.path.join(args.output_dir, "questions.jsonl"), 'w') as f:
    #     for question in questions_all:
    #         f.write(json.dumps(question) +'\n')

