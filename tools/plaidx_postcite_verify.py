import time
import re
import unicodedata
import requests
import json
import argparse
from neuclir_postprocess import *

def normalize_texts(texts):
    texts = unicodedata.normalize('NFKC', texts)
    texts = texts.strip()
    pattern = re.compile(r"\s+")
    texts = re.sub(pattern, ' ', texts).strip()
    pattern = re.compile(r"\n")
    texts = re.sub(pattern, ' ', texts).strip()
    return texts

def get_plaid_response(
    request_id,
    query,
    topk=100, 
    lang='zho', 
    writer=None
):
    url=f"""
        https://trec-neuclir-search.umiacs.umd.edu/query?query={query}&key=allInTheGroove33&content=true&limit={topk}&lang={lang}
    """
    response = requests.get(url.strip())
    response = json.loads(response.content)

    # query_ = response['query']
    system = response['system']
    results = response['results']

    hits = {'doc_ids': [], 'contents': [], 'scores': []}

    for i, result in enumerate(results):
        rank = i+1 

        doc_id = result['doc_id']
        normalized_content = normalize_texts(result['content'])

        hits['doc_ids'].append(doc_id)
        hits['contents'].append(normalized_content)

        if writer is not None:
            writer.write(f"{request_id}\t{lang_id[:2]}\t{rank}\t{doc_id}\t{normalized_content}\n")
            # json.dumps(hits, fout, ensure_ascii=False)

    return hits

def predict_fact_scores(text, evidences):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("-data", "--data_jsonl", type=str, default=None)
    parser.add_argument("-report", "--report_json", type=str, default=None)
    parser.add_argument("-k", "--top_k", type=int, default=100)
    parser.add_argument("-out", "--output_jsonl", type=str, default=None)
    parser.add_argument("-run_id", "--run_id", type=str, default=None)
    # post retrieval setup
    parser.add_argument("--max_word_length", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    # prepare query
    data_items = json.load(open(args.report_json))

    outputs = []
    for item in tqdm(data_items):
        # meta data
        request_id = item['requestid']
        collection_ids = item['collectionids']
        lang_id = collection_ids.split('/')[:2]
        raw_report = item['report']

        # initialize an output placeholder
        output = ReportGenOutput(
            request_id=request_id,
            run_id=args.run_id,
            collection_ids=[item['collection_ids']],
            raw_report=raw_report,
            cited_report=None
        )

        # extract snippets as query for post-hoc search
        snippets = output.get_snippets(max_word_length=args.max_word_length)
        for i, snippet in enumerate(snippets):
            hits = get_plaid_response(
                request_id=request_id,
                query=snippet,
                request_id,
                query,
                topk=args.top_k,
                lang=lang_id,
                writer=None
            )

            ## append only the new document and its contents
            if i == 0:
                hits_all = copy(hits)
            else:
                for doc_id, content in enumerate(
                    zip(hits['doc_ids'], hits['contents'])
                ):
                    if doc_id not in hits_all['doc_ids']:
                        hits_all['doc_ids'] += [docid]
                        hits_all['contents'] += [content]

        output.set_references(hits_all['doc_ids'])

        # rerank the references as citations
        for idx_text, text in enumerate(output.texts):
            scores = predict_fact_scores(text, hits_all['contents'])
            # at most 2 citations
            reference_idx = [i for i in scores.argsort()[::-1][:2] \
                    if scores[i] >= args.fact_threshold ]

            output.set_citations(idx_text, referenceid=reference_idx)
            

        # append the output of a topic)
        outputs.append()


    # prepare writer
    writer = open(args.submission_jsonl, 'w')
    writer.close()

