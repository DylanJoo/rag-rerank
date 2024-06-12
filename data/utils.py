import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def irrelevant_removal(items, ndoc=None, key='full'):
    """ setting criteria to exclude the retrieved top-k 

    items: List of document item. keys include title, text, summary, ...
    """
    to_return = []
    for item in items[:ndoc]:
        ## criteria of inclusion 
        ### 1: text include irrelevant
        ### 2: relevance score less than threshold
        if "irrelevant" in item[key] or "Irrelevant" in item[key]:
            continue
        if ("relevance" in item) and (item["relevance"] < 0.0):
            continue
        else:
            to_return.append(item)

    logger.warn(f"Removed {len(items) - len(to_return)} documents.")
    return to_return

# def get_alce_shorter_text(docs, ndoc, key):
#     doc_list = []
#     ## Option1: original compressed texts
#     for item_id, item in enumerate(docs):
#         if key not in item:
#             if len(doc_list) == 0:
#                 # If there aren't any document, at least provide one (using full text)
#                 item[key] = item['text']
#                 doc_list.append(item)
#             logger.warn(f"No {key} found in document. It could be this data do not contain {key} or previous documents are not relevant. This is document {item_id}. This question will only have {len(doc_list)} documents.")
#             break
#         if "irrelevant" in item[key] or "Irrelevant" in item[key]:
#             continue
#         else:
#             doc_list.append(item)
#         if len(doc_list) >= ndoc:
#             break
#     return doc_list
