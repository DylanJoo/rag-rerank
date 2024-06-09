import argparse
import json

def main(args):

    # eval_data = json.load(open(args.eval_data, 'r'))
    output = json.load(open(args.result_data, 'r'))

    for data in output['data'][:30]:
        print("===== Q:", data['question'].strip(), "=====")
        print("A:", data['answer'].strip())

        claims = ""
        for i, claim in enumerate(data['claims']):
            claims += f"C-{i+1}: {claim.strip()}\n"
        print(claims)

        print("O:", data['output'].strip())

        docs = ""
        for i, doc in enumerate(data['docs']):
            docs += f"D-{i+1}: {doc['title'].strip()}\n"
        if docs == "":
            print('D-0: NO Documents')
        else:
            print(docs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("-r", "--result_data", type=str, default=None)
    args = parser.parse_args()
    main(args)
