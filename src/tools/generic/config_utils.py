import json
import sys
import yaml
import argparse

def load_yaml_config(file_path):
    """Load YAML configuration file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def parse_args(parser, commands):
    """Parse hierarchical command-line arguments."""
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)

    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)

    # Parse base arguments
    parser.parse_args(split_argv[0], namespace=args)

    # Parse subcommands
    for argv in split_argv[1:]:
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)

    return args

def parse_rag_command(commands, yaml_config):
    """Create subparsers and set default values from YAML."""
    data_parser = commands.add_parser("data")
    data_config = yaml_config['data']
    data_parser.add_argument("--index_dir", type=str, default=data_config['index_dir'])
    data_parser.add_argument("--corpus_dir", type=str, default=data_config['corpus_dir'])
    data_parser.add_argument("--topic_file", type=str, default=data_config['topic_file'])
    data_parser.add_argument("--qrels_file", type=str, default=data_config['qrels_file'])
    data_parser.add_argument("--judgement_file", type=str, default=data_config['judgement_file'])
    # crux-specific
    data_parser.add_argument("--n_questions", type=int, default=data_config['n_questions'])
    data_parser.add_argument("--threshold", type=int, default=data_config['threshold'])

    rt_parser = commands.add_parser("retrieval")
    rt_config = yaml_config['retrieval']
    rt_parser.add_argument("--model", type=str, default=rt_config['model'])
    rt_parser.add_argument("--k", type=int, default=rt_config['k'])
    rt_parser.add_argument("--k1", type=float, default=rt_config['k1'])
    rt_parser.add_argument("--b", type=float, default=rt_config['b'])
    rt_parser.add_argument("--batch_size", type=int, default=rt_config['batch_size'])

    rr_parser = commands.add_parser("reranking")
    rr_config = yaml_config['reranking']
    rr_parser.add_argument("--model_class", type=str, default=rr_config['model_class'])
    rr_parser.add_argument("--model_name_or_path", type=str, default=rr_config['model_name_or_path'])
    rr_parser.add_argument("--top_k", type=int, default=rr_config['top_k'])
    rr_parser.add_argument("--batch_size", type=int, default=rr_config['batch_size'])
    rr_parser.add_argument("--max_length", type=int, default=rr_config['max_length'])

    lw_parser = commands.add_parser("listwise_reranking")
    lw_config = yaml_config['listwise_reranking']
    lw_parser.add_argument("--model_name_or_path", type=str, default=lw_config['model_name_or_path'])
    lw_parser.add_argument("--max_k", type=int, default=lw_config['max_k'])
    lw_parser.add_argument("--batch_size", type=int, default=lw_config['batch_size'])
    lw_parser.add_argument("--max_length", type=int, default=lw_config['max_length'])
    lw_parser.add_argument("--use_logits", default=lw_config['use_logits'], action='store_true')
    lw_parser.add_argument("--use_alpha", default=lw_config['use_alpha'], action='store_true')
    lw_parser.add_argument("--num_passes", type=int, default=lw_config['num_passes'])
    lw_parser.add_argument("--system_message", type=str, default=lw_config['system_message'])

    aug_parser = commands.add_parser("augmentation")
    aug_config = yaml_config['augmentation']
    aug_parser.add_argument("--type", type=str, default=aug_config['type'])
    aug_parser.add_argument("--max_k", type=int, default=aug_config['max_k'])
    aug_parser.add_argument("--batch_size", type=int, default=aug_config['batch_size'])
    aug_parser.add_argument("--max_length", type=int, default=aug_config['max_length'])

    gen_parser = commands.add_parser("generation")
    gen_config = yaml_config['generation']
    gen_parser.add_argument("--model_name_or_path", type=str, default=gen_config['model_name_or_path'])
    gen_parser.add_argument("--batch_size", type=int, default=rr_config['batch_size'])
    gen_parser.add_argument("--max_length", type=int, default=gen_config['max_length'])
    gen_parser.add_argument("--temperature", type=float, default=gen_config['temperature'])
    # gen_parser.add_argument("--think_activated", type=bool, action='store_true', default=gen_config['think_activated'])
    return commands

def pretty_print_args(args):
    """Neatly print out parsed arguments."""
    def recursive_namespace_to_dict(namespace):
        """Recursively convert argparse.Namespace to a dictionary."""
        if isinstance(namespace, argparse.Namespace):
            return {key: recursive_namespace_to_dict(value) for key, value in vars(namespace).items()}
        return namespace

    args_dict = recursive_namespace_to_dict(args)
    print(json.dumps(args_dict, indent=4))  # Pretty-print JSON format

if __name__ == "__main__":
    ## Here is the sample calling pipeline
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--default_config", type=str, default="configs/dummy.yaml", help="Path to the configuration YAML file")
    config_args, remaining_argv = config_parser.parse_known_args()
    
    # Step 3: Load the YAML config
    yaml_config = load_yaml_config(config_args.default_config)
    
    # Step 4: Create the main parser
    parser = argparse.ArgumentParser(description="Hierarchical Argument Parser", parents=[config_parser])
    commands = parser.add_subparsers(title="Sub-commands")
    commands = parse_rag_command(commands, yaml_config)

    # Step 5: Parse final arguments
    args = parse_args(parser, commands)
    retty_print_args(args)
