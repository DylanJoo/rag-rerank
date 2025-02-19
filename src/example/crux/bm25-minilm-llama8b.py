import argparse
from tools import load_corpus, load_topics
from tools import load_yaml_config, parse_args, parse_rag_command

def main(args):

    # Data 
    topics = load_topics(args.data.topic_file)
    corpus = load_corpus(args.data.corpus_dir)

    # Retrieval
    from retrieve.bm25 import search
    output_run = search(
        index=args.retrieval.index_dir,
        k1=args.retrieve.k1, 
        b=args.retrieve.b,
        topics=topics,
        batch_size=args.retrieval.batch_size,
        k=args.retrieval.k, 
    )

    # Passage reranking
    from augment.pointwise import rerank
    output_run = rerank(
        topics=topics,
        corpus=corpus,
        runs=output_run,
        reranker_config={
            "model_class": args.reranking.model_class,
            "model_name_or_path": args.reranking.model_name_or_path,
            "device": 'cuda',
            "fp16": True
        },
        top_k=args.top_k,
        batch_size=args.reranking.batch_size,
        max_length=args.reranking.max_length,
    )


    # PROMPT = "Write a passage that answers the given query. Use the provided search results to draft the answer (some of them might be irrelevant). Cite the documents if they are relevant. Write the passage within 100 words. Add the `<p>` and `</p>` tags at the beginning and the end.\n\nQuery: {Q}\nSearch results:\n{Ds}\nPassage: <p>"
    #
    # from generate.llm.hf_back import LLM
    # generator = LLM(model='meta-llama/Llama-3.2-1B-Instruct', temperature=0.7)
    # xs = []
    # for qid in example_topic:
    #     q = example_topic[qid]
    #     ds = output_context[qid]
    #     xs.append(PROMPT.replace("{Q}", q).replace("{Ds}", ds))
    #
    # output_response = generator.generate(x=xs, max_tokens=500)
    # print([r.split('</p>')[0] for r in output_response])

if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--default_config", type=str, default="configs/crux/default.yaml")
    config_args, remaining_argv = config_parser.parse_known_args()
    
    yaml_config = load_yaml_config(config_args.default_config)
    
    parser = argparse.ArgumentParser(description="Hierarchical Argument Parser", parents=[config_parser])
    commands = parser.add_subparsers(title="Sub-commands")
    commands = parse_rag_command(commands, yaml_config)

    args = parse_args(parser, commands)
    pretty_print_args(args)

    main(args)
