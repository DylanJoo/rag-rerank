import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import yaml
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from glob import glob

from llm.base import LLM
from prompts.mds import *
from data_augmentation.utils import normalize_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")

    # Evaluation file is a json file that contains a list of item, each of which contains
    parser.add_argument("--input_dir", type=str, help="Path to generated results")
    parser.add_argument("--multi_news_file", type=str, help="Path to multi-news")
    parser.add_argument("--wcep_10_file", type=str, help="Path to wcep-10")
    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")

    # ICL setting
    parser.add_argument("--ndoc", type=int, help="Number of documents, the exact number will go in decoder.")
    parser.add_argument("--ndoc_pool", type=None, help="Number of documents pool. None will be the same as ndoc")
    parser.add_argument("--shot", type=int, help="Number of ICL demonstrations")
    parser.add_argument("--ndoc_in_demo", type=int, default=None, help="When using --fewer_doc_in_demo, use this to designate how many docs in demo")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")

    # Model and name
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)") # use shard here
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--load_mode", type=str, default='no', help="Model to use")

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, default=1, help="Sample multiple answers.")

    # Use summarization/extraction of the documents
    parser.add_argument("--ampere_gpu", default=False, action='store_true')
    # parser.add_argument("--used_field_in_demo", type=str, default=None, help="Use compressed text data. Option: `full`, `summary`, `extraction`")

    # Load config
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    if "turbo" in args.model:
        args.max_length = 4096
    if "16k" in args.model:
        args.max_length = 16384
    elif "32k" in args.model:
        args.max_length = 32768
    elif "turbo" in args.model:
        args.max_length = 4096
    elif "gpt-4" in args.model:
        args.max_length = 8192
    elif "llama-2" in args.model.lower() or "llama2" in args.model.lower():
        args.max_length = 4096
    elif "llama-3" in args.model.lower() or "llama3" in args.model.lower():
        args.max_length = 8192
    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

    logger.info(f"Set the model max length to {args.max_length} (if not correct, check the code)")

    if args.ndoc_pool is None:
        args.ndoc_pool = args.ndoc
    logger.info(f"Set the model max number of documents to {args.ndoc}/{args.ndoc_pool}")
        
    # Load the model or setup the API
    llm = LLM(args)
    
    # Generate prompts
    np.random.seed(args.seed)

    # Load training data
    # train_data = None

    # Load evaluation data
    from datasets import load_from_disk, concatenate_datasets
    multi_news = load_from_disk(args.multi_news_file)['train'] # hf data
    wcep_10 = load_from_disk(args.wcep_10_file)['train'] # torch files are from original raw file
    ## super long document: multi_news 16430

    ## preproces documents here
    import re
    def normalize(string):
        string = string.strip()
        pattern = re.compile(r"\n")
        string = re.sub(pattern, ' ', string).strip()
        pattern = re.compile(r"\s+")
        string = re.sub(pattern, ' ', string).strip()
        pattern = re.compile("</s>")
        string = re.sub(pattern, '|||||', string).strip() # align seperation 
        return string

    multi_news = multi_news.map(lambda x: {"document": normalize(x['document']), 'mds-source': 'multi_news'})
    wcep_10 = wcep_10.map(lambda x: {"document": normalize(x['document']), 'mds-source': 'wcep-10'})
    eval_dataset = concatenate_datasets([multi_news, wcep_10])

    # Sample quick test
    if args.quick_test is not None:
        np.random.seed(args.seed)
        eval_ids = np.random.choice(len(eval_dataset), args.quick_test, replace=False)
        eval_dataset = [eval_dataset[int(idx)] for idx in eval_ids]

    # build mapping for full text
    fulltexts = {}
    logger.info("Build full text mapping...") 
    for idx, eval_item in enumerate(tqdm(eval_dataset)):
        example_id = f"{eval_item['mds-source']}-{eval_ids[idx]}"
        document_list = eval_item['document'].split('|||||')
        document_list = [normalize_texts(d, 5000) for d in document_list]
        fulltexts[example_id] = {'mds': eval_item['summary'], 'docs': document_list}
    logger.info("Done full text mapping.")

    # build mapping for questions and summaries
    logger.info("load questions...") 
    from data_augmentation.utils import load_question
    questions_all = []
    for file in tqdm(glob(os.path.join(args.input_dir, "*ques*.json"))):
        questions = load_question(file)
        questions_all += questions
    questions_all = {q['example_id']: q['texts'] for q in questions_all}

    logger.info("load passages...") 
    from data_augmentation.utils import load_passages
    passages_all = []

    for file in tqdm(glob(os.path.join(args.input_dir, "*summ*.json"))):
        passages = load_passages(file)
        passages_all += passages
    passages_all = {p['example_id']: p['texts'] for p in passages_all}

    ## get intercept
    overlap = questions_all.keys() & passages_all.keys()
    questions_all = {k: v for k, v in questions_all.items() if k in overlap}
    passages_all = {k: v for k, v in passages_all.items() if k in overlap}
    logger.info(f"{len(questions_all)} remained...")

    # Start generation
    logger.info("Generating output...")

    ratings = []
    for t, example_id in enumerate(tqdm(questions_all)):
        if t == 10:
            break
        # mds = fulltexts[example_id]['mds']
        questions = questions_all[example_id]
        docs = fulltexts[example_id]['docs']
        passages = passages_all[example_id]

        output_array = []
        for i, passage_list in enumerate(passages):
            document = docs[i]

            for j, passage in enumerate(passage_list):
                if len(passage) > 10:
                    output_vector = [-1 for _ in questions]
                    for k, question in enumerate(questions):
                        prompt = prompt_rating_gen(
                            INST=instruction_rating,
                            Q=question,
                            C=passage,
                            PREFIX="Rating:"
                        )
                        prompt_len = len(llm.tokenizer.tokenize(prompt))
                        output = llm.generate(prompt, min(args.max_new_tokens, args.max_length-prompt_len))
                        output = output.replace("<|im_end|>", "").rstrip()
                        if output.endswith("End."):
                            output = output[:-len("End.")]

                        # extract rating
                        c = re.compile(r"\d|-\d")
                        output = re.findall(c, output + "-1")[0]
                        output = -1 if len(output) == 0 else int(output)
                        output_vector[k] = output
                else:
                    output_vector = [-1 for _ in questions]
                    prompt = ""
                    prompt_len = -1

                output_array.append(output_vector)
                logger.info(f"Example: {example_id} - {i} - {j}")
                logger.info(f"prompt text (length={prompt_len}): {prompt}")
                logger.info(f"Final model output: {output_vector}") 

        ratings.append({
            "example_id": example_id,
            "passages": passages,
            "questions": questions,
            "ratings": output_array
        })

    # Save the result
    name = "question_to_summaries_llama3.1"

    if not os.path.exists("data/mds/alignment"):
        os.makedirs("data/mds/alignment")

    with open("data/mds/alignment/" + name + ".jsonl", "w") as f:
        for rating in ratings:
            f.write(json.dumps(rating, indent=4)+'\n')

if __name__ == "__main__":
    main()

