import re

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def postprocess(sent, tag='p'):
    sent = remove_citations(sent)
    sent = sent.strip().split(f'</{tag}>')[0]
    return sent
