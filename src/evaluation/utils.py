import os
import re
import argparse
import json
import numpy as np
from copy import copy
import random
from collections import defaultdict
from tqdm import tqdm
from glob import glob


# def load_questions(path, n=10):
#     questions = {}
#     with open(path, 'r') as f:
#         for line in f:
#             item = json.loads(line.strip())
#             example_id = item['example_id']
#             questions[example_id] = [replace_tags(q) for q in item['questions']][:n]
#     return questions
#
# def replace_tags(sent, tag='q'):
#     if tag == 'q':
#         sent = re.sub(r"\<q\>|\<\/q\>", "\n", sent)
#     if tag == 'p':
#         sent = re.sub(r"\<p\>|\<\/p\>", "\n", sent)
#     if tag == 't':
#         sent = re.sub(r"\<r\>|\<\/r\>", "\n", sent)
#     pattern = re.compile(r"\n+")
#     sent = re.sub(pattern, '\n', sent)
#     pattern = re.compile(r"^(\d+)*\.")
#     sent = re.sub(pattern, '', sent)
#     return sent
#

# def load_passages(dir, report_file=None):
#     passages = {}
#     for file in glob(os.path.join(dir, "*jsonl")):
#         with open(file, 'r') as f:
#             for line in f:
#                 item = json.loads(line.strip())
#                 passages[item["id"]] = item['contents']
#
#     if report_file is not None:
#         with open(report_file, 'r') as f:
#             for line in f:
#                 item = json.loads(line.strip())
#                 example_id = item['example_id']
#                 passage[example_id] = item['contents']
#     return passages

# def load_contexts(path, report_file=None):
#     contexts = {}
#     with open(path, 'r') as f:
#         for line in f:
#             item = json.loads(line.strip())
#             contexts[item["id"]] = item['contents']
#     return contexts
