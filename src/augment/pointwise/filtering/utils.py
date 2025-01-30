
def apply_docs_prompt(texts, titles, ndoc=None):
    p = ""
    assert len(texts) == len(titles), 'inconsistent length of texts and titles.'
    for idx, (text, title) in enumerate(zip(texts[:ndoc], titles)):
        p_doc = "[{ID}]:{T}{P}\n"
        p_doc = p_doc.replace("{ID}", str(idx+1))
        if title == "":
            p_doc = p_doc.replace("{T}", "")
        else:
            p_doc = p_doc.replace("{T}", f" (Title: {title}) ")
        p_doc = p_doc.replace("{P}", text)
        p += p_doc
    return p

