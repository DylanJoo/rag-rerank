import unicodedata
import requests
import json

def get_neuclir_response(qid='0', query='helloworld', topk=10, lang='zho'):
    url=f"""
        https://trec-neuclir-search.umiacs.umd.edu/query?query='{query}'key=allInTheGroove33&content=true&limit={topk}&lang={lang}
    """
    response = requests.get(url.strip())
    response = json.loads(response.content)

    query_ = response['query']
    system = response['system']
    results = response['results']

    to_return = {
            'qid': qid, 'query': query, 
            'docid': [], 'content': [], 'scores': []
    }

    for result in results:
        to_return['docid'].append(result['doc_id'])
        normalized_content = unicodedata.normalize('NFKC', result['content'])
        to_return['content'].append(normalized_content)

    return to_return
