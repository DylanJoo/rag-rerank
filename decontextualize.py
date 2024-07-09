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
from prompts.mds import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--shard", type=int, default=0, help="the n-th shard")
    parser.add_argument("--shard_size", type=int, default=200, help="size of one shard")
    parser.add_argument("--generation", type=str, default='claims', choices=['claims', 'question'])

    # Evaluation file is a json file that contains a list of item, each of which contains
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
    # parser.add_argument("--used_field", type=str, default="full", help="Use compressed text data. Option: `full`, `summary`, `extraction`")
    # parser.add_argument("--used_field_in_demo", type=str, default=None, help="Use compressed text data. Option: `full`, `summary`, `extraction`")

    # Load config
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

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
    logger.info(f"Set the model max length to {args.max_length} (if not correct, check the code)")

    if args.ndoc_pool is None:
        args.ndoc_pool = args.ndoc
    logger.info(f"Set the model max number of documents to {args.ndoc}/{args.ndoc_pool}")
        
    # Load the model or setup the API
    llm = LLM(args)
    
    # Generate prompts
    np.random.seed(args.seed)

    # Load training data
    train_data = None

    # Load evaluation data
    from datasets import load_from_disk, concatenate_datasets
    multi_news = load_from_disk(args.multi_news_file)['train'] # hf data
    wcep_10 = load_from_disk(args.wcep_10_file)['train'] # torch files are from original raw file

    ## preproces documents here
    import re
    def normalize(string):
        string = string.strip()
        pattern = re.compile(r"\s+")
        string = re.sub(pattern, ' ', string).strip()
        pattern = re.compile(r"\n")
        string = re.sub(pattern, '', string).strip()
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

    # Generate the demonstration part
    if args.shot > 0:
        pass
        # train_ids = np.random.choice(len(train_data["demos"]), args.shot, replace=False)
        # logger.info("Generating demo (in-context) example...") 
        # demo_prompt = ""
        # for i, train_id in enumerate(train_ids):
        #
        #     train_item = train_data["demos"][train_id]
        #     doc_prompt = apply_docs_prompt(
        #         doc_items=train_item["docs"], 
        #         ndoc=args.ndoc_in_demo,
        #         field=(args.used_field_in_demo or args.used_field)
        #     )
        #     demo_prompt += apply_demo_prompt(
        #         Q=train_item["question"],
        #         D=doc_prompt, 
        #         A=train_item["answer"],
        #         instruction="" # if adding instruction for each demo examples
        #     )
        #     demo_prompt += demo_sep
    else:
        demo_prompt = ""

    # Generate the input prompt part
    eval_data = []
    eval_documents = {}
    logger.info("Generating prompts...") 
    for idx, eval_item in enumerate(tqdm(eval_dataset)):

        ## preprocess for claim generation of sumamry
        if args.generation == 'claims':
            prompt = apply_inst_prompt_claim_gen(
                Q="",
                D=eval_item['summary'],
                instruction=instruction_prompt_c,
                add_prefix=True
            )
        if args.generation == 'question':
            prompt = apply_inst_prompt_question_gen(
                Q="",
                D=eval_item['summary'],
                instruction=instruction_prompt_q,
                add_prefix=True
            )
        prompt = prompt.replace("{DEMO}", demo_prompt)

        eval_data.append({
            'example_id': f"mds-{args.generation}_{args.shard}-{idx}", 
            'mds-source': eval_item['mds-source'],
            'prompt': prompt,
            'full_text': eval_item['summary']
        })

        if args.generation == 'claims':
            document_list = eval_item['document'].split('|||||')
        else:
            document_list = []
        eval_documents[idx] = {'prompts': [], 'full_texts': document_list}

        ## preprocess for claim generation of doc
        for doc_idx, doc_text in enumerate(document_list):
            prompt = apply_inst_prompt_claim_gen(
                Q="",
                D=doc_text,
                instruction=instruction_prompt_c,
                add_prefix=True
            )
            prompt = prompt.replace("{DEMO}", demo_prompt)
            eval_documents[idx]['prompts'].append(prompt)
    logger.info("Done prompt preparation.")

    # Start generation
    start = args.shard * args.shard_size
    end = start + args.shard_size
    if start >= len(eval_data):
        exit(0) # finished

    eval_data = eval_data[start:end]
    for idx, item in enumerate(tqdm(eval_data)):
        # summary claims
        prompt = item['prompt']
        prompt_len = len(llm.tokenizer.tokenize(prompt))
        num_docs = len(eval_documents[idx]['prompts'])
        full_text = item.pop('full_text')

        if idx == 0:
            print(prompt)

        ## only log the metadata of sumamry. The other claims of documetns are not.
        output_array = [llm.generate(prompt, min(args.max_new_tokens, args.max_length-prompt_len))]
        item['prompt_len'] = prompt_len
        item['number_of_documents'] = num_docs
        
        output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
        if output_array[-1].endswith("End."):
            output_array[-1] = output_array[-1][:-len("End.")]

        logger.info(f"Prompt text (length={prompt_len}): {prompt}")
        logger.info(f"Final model output (summary's): {output_array[-1]}") 
        logger.info(f"Number of documents {num_docs}") 

        ### if we dont have a good-ish summary's claim. we move on to the next
        if (len(output_array[-1].split('[')) < 2) and (args.generation == 'claims'): # about a size of a claim
            item['output'] = []
            logger.info("Bypass this claims of documents', since the summary's is not successful")
        else:
            # document claims
            for prompt in eval_documents[idx]['prompts']:
                output_array.append(llm.generate(prompt, min(args.max_new_tokens, args.max_length-prompt_len)))
                
                output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
                if output_array[-1].endswith("End."):
                    output_array[-1] = output_array[-1][:-len("End.")]

            item['full_text'] = [full_text] + eval_documents[idx]['full_texts']
            item['output'] = output_array if len(output_array) > 1 else output_array[0]
        
    # logger.info(f"#Cases when prompts exceed max length: {llm.prompt_exceed_max_length}")
    # logger.info(f"#Cases when max new tokens < 50: {llm.fewer_than_50}")

    # Save the result
    model_name = args.model
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    name = f"{args.dataset_name}-{model_name}-{args.shard}-{args.tag}-0shotx{args.ndoc_in_demo}-{args.seed}"
    # name = f"{args.dataset_name}-{model_name}-{args.shard}-{args.tag}-0shot-{args.seed}"

    if args.quick_test is not None:
        name += f"-quick_test{args.quick_test}"

    eval_data = {"args": args.__dict__, "data": eval_data}

    if not os.path.exists("data/mds"):
        os.makedirs("data/mds")
    json.dump(eval_data, open("data/mds/" + name + ".json", "w"), indent=4)

if __name__ == "__main__":
    main()

