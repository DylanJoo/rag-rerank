import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import re
import os
import yaml
import argparse
import json
import numpy as np
from tqdm import tqdm
from glob import glob

from prompts.mds import *
from evaluation.llm_judge.utils import (
    load_questions, 
    load_contexts, 
    load_judgements
)
def rac_evaluate(
    output_contexts,
):
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--topic_question_file", type=str, help="Path to retrieval-augmented context")
    parser.add_argument("--output_file", type=str, help="Path to judged (graded) retrieval-augmented context")
    parser.add_argument("--context_file", type=str, help="Path to contexts")

    # Load config
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

    logger.info(f"Set the model max length to {args.max_length} (if not correct, check the code)")

    if args.load_mode == 'vllm':
        from generate.llm.vllm_back import vLLM
        llm = vLLM(args)
    else:
        from generate.llm.hf_back import LLM
        llm = LLM(args)

    # load questions, load_run
    questions_all = load_questions(args.topic_question_file, args.n_questions)
    contexts = load_contexts(args.context_file)

    judgements = load_judgements(args.output_file)
    logger.info(f"Total number of judged pairs {len(judgements)} / {len(contexts)}") 

    for psgid in tqdm(contexts, desc=f'Evaluating context ', total=len(contexts)):

        example_id = psgid.split(":")[0]

        # skip the one that have already been done
        if example_id not in questions_all:
            continue 

        questions = questions_all[example_id]
        context = contexts[psgid]

        output = ""
        output_vector = [-1 for _ in questions]

        # skip the one that have already been done
        if judgements[example_id][psgid] is not None:
            continue 

        ## no batch here
        for k, question in enumerate(questions):
            prompt = prompt_rating_gen(
                INST=instruction_rating,
                Q=question,
                C=context,
                PREFIX="Rating:"
            )
            output = llm.generate(
                prompt, 
                max_tokens=args.max_new_tokens,
                min_tokens=1
            )
            output = [o.replace("<|im_end|>", "").rstrip() for o in output][0]

            # extract rating
            pattern = re.compile(r"\d|-\d")
            output = re.findall(pattern, output + "-1")[0]
            output = -1 if len(output) == 0 else int(output)
            output_vector[k] = output

        # append on output array
        judgements[example_id][psgid] = output_vector
        logger.info(f"Final model output: {output_vector}") 

    # Save the result
    with open(args.output_file, "w") as f:
        for example_id in judgements:
            for psgid in judgements[example_id]:
                rating = judgements[example_id][psgid]
                f.write(json.dumps({"example_id": example_id, "pid": psgid, "rating": rating})+'\n')

if __name__ == "__main__":
    main()

