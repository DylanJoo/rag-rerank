import json

qids = []
with open('/home/dju/rag-rerank/src/results/-1_back/test-splade-v3_100-vanilla_10.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        qids.append( int(data['qid'].replace('multi_news-test-', "")) )

print(qids)
print(len(qids))
