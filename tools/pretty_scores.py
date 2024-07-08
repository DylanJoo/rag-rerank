from glob import glob
import argparse
import json

KEYS = ['citation_rec', 'citation_prec', 'claims_nli']

def main(args):
    output_dict = {}
    writer = open('scores_all.csv', 'w')

    for file in glob(f'{args.score_dir}/{args.c}.json.score'):
        score_dict = json.load(open(file))
        scores = [score_dict[k] for k in KEYS]
        scores = ", ".join(scores)
        writer.write(f"{file}, {key}, {scores}\n")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("-s", "--score_dir", type=str, default=None)
    parser.add_argument("-c", "--control", type=str, default='*')
    args = parser.parse_args()
    main(args)
