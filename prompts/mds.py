# example_doc = "National Archives\nYes, it’s that time again, folks. It’s the first Friday of the month, when for one ever-so-brief moment the interests of Wall Street, Washington and Main Street are all aligned on one thing: Jobs. \n \n A fresh update on the U.S. employment situation for January hits the wires at 8:30 a.m. New York time offering one of the most important snapshots on how the economy fared during the previous month. Expectations are for 203,000 new jobs to be created, according to economists polled by Dow Jones Newswires, compared to 227,000 jobs added in February. The unemployment rate is expected to hold steady at 8.3%. \n \n Here at MarketBeat HQ, we’ll be offering color commentary before and after the data crosses the wires. Feel free to weigh-in yourself, via the comments section. And while you’re here, why don’t you sign up to follow us on Twitter. \n \n Enjoy the show."

# instruction_prompt_mds = "Instruction: write 3 to 5 comprehensive statements that support the given document below. Each statement should be standalone and provide enough context to be understood independently. These statements should cover the document as thoroughly as possible. Each statement should begin with a number in square brackets (e.g., [1])."

template_mds = "Instruction: {INST}\n\nDocument: {D}\n\n{PREFIX}"
instruction_statement = "Write 10 diverse statements that are evident from the given document. Each statement should be standalone and have necessary context. Every statements should start with the format of a square bracket with number (e.g., [1])."
def prompt_statement_gen(INST="", D="", PREFIX="Statements:\n[1]: "):
    p = template_mds
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{D}", D)
    p = p.replace("{PREFIX}", PREFIX).strip()
    return p

template_doc = "Instruction: {INST}\n\nDocument: {D}\n\n{PREFIX}"
# instruction_summary = "Break down the given document into two standalone passages."
instruction_summary = "Break down the given document into two standalone passages. Each passages should be self-contained and have necessary context. Keep the wording as similar as possible to the original document. These passages should start with the format of a square bracket with number (e.g., [1] and [2])."
# instruction_summary = "Write two standalone summaries for the given document."
def prompt_summary_gen(INST="", D="", PREFIX="Summaries:"):
    p = template_doc
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{D}", D)
    p = p.replace("{PREFIX}", PREFIX).strip()
    return p


# instruction_summary = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided documents as references (some of which might be irrelevant). Always cite at least one document for every sentences in the answer. Use the citation format of square brackets to indicate the cited documents (e.g., [1] for the first reference). If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. Only generate the answer, excluding any disclaimers, notes or list of references at the end of the answer."
def apply_docs_prompt(doc_items, ndoc=None, field='text'):
    p = ""
    for idx, doc_item in enumerate(doc_items[:ndoc]):
        p_doc = doc_prompt_template
        p_doc = p_doc.replace("{ID}", str(idx+1))
        p_doc = p_doc.replace("{T}", doc_item['title'])
        p_doc = p_doc.replace("{P}", doc_item[field])
        p += p_doc
    return p

