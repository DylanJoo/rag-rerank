import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import yaml
import argparse
import json
import numpy as np
from tqdm import tqdm

from llm.base import LLM
from prompts.eli5 import *
from data.utils import get_alce_shorter_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file")

    # Evaluation file is a json file that contains a list of item, each of which contains
    parser.add_argument("--eval_file", type=str, help="Path to the eval file")
    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")

    # ICL setting
    parser.add_argument("--ndoc", type=int, help="Number of documents")
    parser.add_argument("--shot", type=int, help="Number of ICL demonstrations")
    parser.add_argument("--ndoc_in_demo", type=int, default=None, help="When using --fewer_doc_in_demo, use this to designate how many docs in demo")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument("--retrieve_in_all_docs", type=bool, default=False, help="Retrieve in all documents instead of just top ndoc")

    # Model and name
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--load_mode", type=str, default='int4', help="Model to use")

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, default=1, help="Sample multiple answers.")

    # Use summarization/extraction of the documents
    parser.add_argument("--use_shorter", type=str, default=None, help="Use compressed text data. Option: None, `summary`, `extraction`")

    # Interactive
    # parser.add_argument("--interactive", type=bool, default=False, help="Whether to run in interactive mode")
    # parser.add_argument("--interactive_query", type=str, default=None, help="The query to use in interactive mode, either `doc_id` (corresponding to interact in paper) or `search` (corresponding to inlinesearch in paper).")
    # parser.add_argument("--retriever", type=str, default=None, help="When using interactive search mode, which retriever to use. Options: `tfidf`, `gtr-t5-large`")
    # parser.add_argument("--retriever_device", type=str, default="cuda", help="Where to put the dense retriever if using. Options: `cuda`, `cpu`")
    # parser.add_argument("--max_turn", type=int, default=10, help="Max number of all actions")
    # parser.add_argument("--max_doc_show", type=int, default=3, help="Max number of documents to show at one time.")
    # parser.add_argument("--force_cite_show", type=bool, default=False, help="Force citing the documents that are shown to the model")

    # Load config
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

    if "turbo" in args.model:
        # ChatGPT has a longer max length
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
    logger.info(f"Set the model max length to {args.max_length} (if not correct, check the code)")
        
    # Load the model or setup the API
    llm = LLM(args)
    
    # Generate prompts
    np.random.seed(args.seed)

    # Load data
    prompt_data = json.load(open(args.prompt_file))
    eval_data = json.load(open(args.eval_file))

    # Generate the demonstration part
    head_prompt = ""
    train_ids = np.random.choice(len(prompt_data["demos"]), args.shot, replace=False)

    ## Prepare instruction and the demo prompts
    #### in-context demo examepls via retrieved document
    demo_prompt = ""
    for i, train_id in enumerate(train_ids):

        train_item = prompt_data["demos"][train_id]
        doc_prompt = apply_docs_prompt(
            doc_items=train_item["docs"], 
            ndoc=args.ndoc_in_demo,
            field='summary'
        )
        demo_prompt += apply_demo_prompt(
            Q=train_item["question"],
            D=doc_prompt, 
            A=train_item["answer"],
            instruction="" # if adding instruction for each demo examples
        )
        demo_prompt += demo_sep

    # Sample quick test
    if args.quick_test is not None:
        np.random.seed()
        eval_ids = np.random.choice(len(eval_data), args.quick_test, replace=False)
        eval_data = [eval_data[int(idx)] for idx in eval_ids]

    logger.info("Generating prompts...") 
    incomplete_doc_list = 0 # For some questions there might be fewer than ndoc documents
    for idx, eval_item in enumerate(tqdm(eval_data)):

        doc_prompt = apply_docs_prompt(
            eval_item["docs"], 
            args.ndoc, 
            'summary'
        )
        ## option1: use only one system prompt to instruct
        prompt = apply_inst_prompt(
            Q=eval_item['question'],
            D=doc_prompt,
            instruction=instruction_prompt,
            add_prefix=True
        )
        prompt = prompt.replace("{DEMO}", demo_prompt)

        ## option2: use multiple instruction prompts as in-context examples
        # prompt = apply_demo_prompt(
        #     Q=eval_item['question'],
        #     D=doc_prompt,
        #     instruction=system_prompt,
        #     add_prefix=True
        # )

        eval_data[idx]['prompt'] = prompt
        doc_list = get_alce_shorter_text(
            eval_item, 
            eval_item["docs"], 
            args.ndoc, 
            args.use_shorter
        ) if args.use_shorter is not None else eval_item["docs"][:args.ndoc]
        if not args.retrieve_in_all_docs:
            # If --retrieve_in_all_docs, we keep the original docs and do not trim them by ndoc
            # Otherwise, take the new docs (truncated by ndoc and filtered if using summary/extraction)
            eval_data[idx]['docs'] = doc_list
        if len(doc_list) < args.ndoc:
            incomplete_doc_list += 1
    logger.info("Done.")
    if incomplete_doc_list > 0:
        logger.warning(f"There are {incomplete_doc_list} questions that have incomplete document list (may due to a lot of them are filtered out by summary/extraction).")

    for idx, item in enumerate(tqdm(eval_data)):
        prompt = item['prompt']
        prompt_len = len(llm.tokenizer.tokenize(prompt))

        if idx == 0:
            print(prompt)

        output_array = []
        for _ in range(args.num_samples):
            output_array.append(llm.generate(prompt, min(args.max_new_tokens, args.max_length-prompt_len)))
            item['prompt'] = prompt
            
            output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
            if output_array[-1].endswith("End."):
                output_array[-1] = output_array[-1][:-len("End.")]

            logger.info(f"Question: {item['question']}")
            logger.info(f"Prompt text (length={prompt_len})={prompt}")
            # logger.info(f"Gold answer: {item['answer']}")
            logger.info(f"Final model output: {output_array[-1]}") 
        
        item['output'] = output_array if len(output_array) > 1 else output_array[0]
        
    logger.info(f"#Cases when prompts exceed max length: {llm.prompt_exceed_max_length}")
    logger.info(f"#Cases when max new tokens < 50: {llm.fewer_than_50}")

    # Save the result
    model_name = args.model
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    name = f"{args.dataset_name}-{model_name}-{args.tag}-shot{args.shot}x{args.ndoc_in_demo}-ndoc{args.ndoc}-{args.seed}"

    if args.quick_test is not None:
        name += f"-quick_test{args.quick_test}"
    # if args.num_samples > 1:
    #     name += f"-sample{args.num_samples}"
    # if args.force_cite_show:
    #     name += f"-forceciteshow"

    eval_data = {"args": args.__dict__, "data": eval_data}

    if not os.path.exists("result"):
        os.makedirs("result")
    json.dump(eval_data, open("result/" + name + ".json", "w"), indent=4)

if __name__ == "__main__":
    main()

