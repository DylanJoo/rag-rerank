
def citation(documents):
    texts = [doc['text'] for doc in documents]
    titles = [doc['title'] for doc in documents]

    p = ""
    assert len(texts) == len(titles), 'inconsistent len of texts and titles.'
    for idx, (text, title) in enumerate(zip(texts, titles)):
        p_doc = "[{ID}] {T}{P}\n"
        p_doc = p_doc.replace("{ID}", str(idx+1))
        if title == "":
            p_doc = p_doc.replace("{T}", "")
        else:
            p_doc = p_doc.replace("{T}", f" (Title: {title}) ")
        p_doc = p_doc.replace("{P}", text)
        p += p_doc
    return p


template_fn_mapping = {
    "citation": citation
}

