import numpy as np
import random
import ast
from tools import load_qrels, load_judgements

def collect_results(file_path, threshold=3):
    qrels = load_qrels("/home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt")
    judgements = load_judgements(f"/home/dju/datasets/crux/ranking_{threshold}/testb_oracle-passages_judgements.jsonl")
    qids = list(qrels.keys())

    random.seed(327)
    selected = [qids[i] for i in random.sample(range(50), 10)]
    random.seed(327)
    print(sorted(random.sample(range(50), 10)))

    results = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            if ('rating' in line): 
                if qids[i] in selected:
                    values_str = line.split(":")[1].strip().strip("[]")
                    rating_list = [int(x) for x in values_str.split()]
                    judgement_oracle = np.array([judgements[qids[i]][docid] for docid in qrels[qids[i]].keys()]).max(0)
                    answerable = (judgement_oracle >= threshold)
                    answerability = [1 for i in range(len(rating_list)) if (answerable[i] and rating_list[i] >= 3)]
                    results.append( sum(answerability) / sum(answerable) )

                i += 1
    return results

oracle_file = "/home/dju/rag-rerank/src/logs/meta-llama/Llama-3.1-70B-Instruct/testb-oracle_k.log"
bm25_file = "/home/dju/rag-rerank/src/logs/meta-llama/Llama-3.1-70B-Instruct/rag_-1/testb-bm25_100-vanilla_-1.log"
dr_rankfirst_file = "/home/dju/rag-rerank/src/logs/meta-llama/Llama-3.1-70B-Instruct/rag_-1/testb-contriever_100-rankfirst_100-vanilla_-1.log"
oracle = collect_results(oracle_file, 3)
bm25 = collect_results(bm25_file, 3)
dr_rankfirst = collect_results(dr_rankfirst_file, 3)
print("oracle: ", oracle, np.mean(oracle))
print("bm25: ", bm25, np.mean(bm25))
print("dr_rankfirst: ", dr_rankfirst, np.mean(dr_rankfirst))

