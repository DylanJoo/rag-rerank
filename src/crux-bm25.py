import argparse
from tools import load_corpus, load_topics, load_judgements, load_qrels
from tools import load_yaml_config, parse_args, parse_rag_command

def main(args):

    # Data 
    topics = load_topics(args.data.topic_file, args.debug)
    corpus = load_corpus(args.data.corpus_dir)
    qrels = load_qrels(args.data.qrels_file)
    judgements = load_judgements(args.data.judgement_file) \
            if args.data.judgement_file is not None else None

    # Retrieval
    from retrieve.bm25 import search
    output_run = search(
        index=args.data.index_dir,
        k1=args.retrieval.k1, 
        b=args.retrieval.b,
        topics=topics,
        batch_size=args.retrieval.batch_size,
        k=args.retrieval.k, 
    )
    print(output_run)

    # Pointiwse reraning
    if args.reranking is not None:
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
            top_k=args.reranking.top_k,
            batch_size=args.reranking.batch_size,
            max_length=args.reranking.max_length,
        )

    # Listiwse reranking 
    if args.listwise_reranking is not None:
        from augment.listwise import rerank
        output_run = rerank(
            topics=topics,
            corpus=corpus,
            runs=output_run,
            model_path=args.listwise_reranking.model_name_or_path,
            top_k=args.listwise_reranking.max_k,
            num_passes=args.listwise_reranking.num_passes,
            prompt_mode='rank_GPT',  # maybe also this parameter
            context_size=4096,       # add this parameter
            use_logits=args.listwise_reranking.use_logits, 
            num_gpus=args.num_gpus, # check if it can be adjusted dynamically
            batch_size=args.listwise_reranking.batch_size,
            use_alpha=args.listwise_reranking.use_alpha,
            vllm_batched=True,
            variable_passages=False,
            window_size=20,
            system_message=args.listwise_reranking.system_message
        )

    # Context augmentation
    from augment.base import vanilla
    output_rac = vanilla(
        topics=topics,
        corpus=corpus,
        runs=output_run,
        max_k=args.augmentation.max_k if args.augmentation else None
    )

    # Retrieval-augmented context evaluation
    if judgements:
        from evaluation import rac_evaluate
        output_eval = rac_evaluate(
            corpus=corpus,
            qrels=qrels, 
            judgements=judgements,
            rac_data=output_rac,
            n_questions=args.data.n_questions,
            threshold=args.data.threshold,
            runs=output_run,
        )
        print(output_eval)

    # Generation
    if args.generation is not None:
        PROMPT = \
        "Write a passage for given query. Always use the provided contexts to write the passage (some of them might be irrelevant). " + \
        "Cite at least one context in each sentence in the passage. When citing several search results, use [1][2][3]. " + \
        "Write the passage within 300 words.\n\nQuery: {Q}\nContexts:\n{Ds}\nPassage: <think>\n"

        from generate.llm.hf_back import LLM
        generator = LLM(
            model=args.generation.model_name_or_path, 
            temperature=args.generation.temperature,
        )
        xs = []
        for qid in topics:
            q = output_rac[qid]['topic']
            ds = output_rac[qid]['prompt']
            xs.append(PROMPT.replace("{Q}", q).replace("{Ds}", ds))

            response = generator.generate(x=xs, max_tokens=args.generation.max_length)
            output_rac[qid]['response'] = response
            print("\n\n".join(response))

if __name__ == "__main__":
    from tools import pretty_print_args
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--default_config", type=str, default=None)
    config_parser.add_argument("--debug", type=int, default=None)
    config_parser.add_argument("--num_gpus", type=int, default=1)
    config_args, remaining_argv = config_parser.parse_known_args()
    
    yaml_config = load_yaml_config(config_args.default_config)
    config_parser.set_defaults(default_config=config_args.default_config)
    
    parser = argparse.ArgumentParser(description="Hierarchical Argument Parser", parents=[config_parser])
    commands = parser.add_subparsers(title="Sub-commands")
    commands = parse_rag_command(commands, yaml_config)

    args = parse_args(parser, commands)
    pretty_print_args(args)

    main(args)
