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

## preproces documents here
import re
def normalize(string):
    string = string.strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    return string

def prep_multi_news(examples):
    for doc in examples['document']:
        examples['docs'] = [normalize(d) for d in doc.split('|||||')]
    return examples

def prep_wcep_10(x):
    for ex in examples:
        examples['docs'] = [normalize(d) for d in doc.split('</s>')]
    return examples

# Load evaluation data
from datasets import load_from_disk, concatenate_datasets
multi_news_file='/home/dju/datasets/multi_news'
wcep_10_file='/home/dju/datasets/wcep-10'
multi_news = load_from_disk(multi_news_file)['train'] # hf data
wcep_10 = load_from_disk(wcep_10_file)['train'] # torch files are from original raw file
print(multi_news)
print(wcep_10)

def normalize(string):
    string = string.strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, '', string).strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, '', string).strip()
    pattern = re.compile("</s>")
    string = re.sub(pattern, '|||||', string).strip() # align seperation 
    return string

multi_news = multi_news.map(lambda x: {"document": normalize(x['document'])})
wcep_10 = wcep_10.map(lambda x: {"document": normalize(x['document'])})
eval_dataset = concatenate_datasets([multi_news, wcep_10])

for d in eval_dataset:
    document_list = d['document'].split('|||||')
