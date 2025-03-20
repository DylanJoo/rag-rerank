import os
import json
import argparse
from tqdm import tqdm

from tools import load_runs, load_corpus, load_topics, load_questions
from augment.template import template_fn_mapping

def vanilla(
    topics, corpus, runs, questions,
    max_k,
    template_type='citation',
    writer=None,
):

    qids = list(topics.keys())
    qids = [qid for qid in qids if qid in runs]

    outputs = {}
    for qid in tqdm(qids, total=len(qids)):

        result = runs[qid]
        topic = topics[qid]
        list_questions = questions[qid]
        documents = [corpus[docid] for docid in result][:max_k]
        raw_content = [(d['title'] + " " + d['text']).strip() for d in documents]

        # arrange
        output = {
            "qid": qid, "topic": topic, "questions": list_questions,
            "type": f"vanilla_{max_k}",
            "docids": [docid for docid in result][:max_k], 
            "context_list": raw_content, 
            "prompt": template_fn_mapping[template_type](documents),
            "report": None,
            "response": None
        }
        outputs[qid] = output

        if writer is not None:
            writer.write(json.dumps(output, ensure_ascii=False)+'\n')

    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic_file", type=str, default=None)
    parser.add_argument("--corpus_dir_or_file", type=str, default=None)
    parser.add_argument("--run_file", type=str, default=None)
    parser.add_argument("--max_k", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)
    writer = open(args.output, 'w')

    ## load data
    topics = load_topics(args.topic_file)
    corpus = load_corpus(args.corpus_dir_or_file)
    runs = load_runs(args.run_file, topk=args.max_k, output_score=True)
    questions = load_questions(args.topic_file)

    vanilla(
        topics=topics, corpus=corpus, run=runs,
        questions=questions,
        max_k=args.max_k,
        template_type=args.template_type,
        writer=writer
    )
    writer.close()

    print('done')
