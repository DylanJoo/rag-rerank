import argparse
import json


promtps = """
Generate a searching query for the given request from an user. The query will serve as a keyword pair to the search engine.
The user's background is also provided as reference. The query should be able to reflect comprehensive information need from the request itself and also the user background.
""".strip()

promtps = """
Generate a search query based the given request from the user. The user's background is also provided for the reference.
""".strip()

def main(args):

    with open(args.input_jsonl, 'r') as file:
        for line in tqdm(file):
            item = json.loads(line.strip())
            item['requrest_id'] = 
            item['collection_ids'] = 
            background = item['background'] 
            problem = item['problem_statement'] 
            item['limit'] = 

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
    parser.add_argument("-input", "--input_jsonl", type=str, default=None)
    parser.add_argument("-output", "--output_csv", type=str, default=None)
    args = parser.parse_args()
    main(args)
