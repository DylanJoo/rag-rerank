# instruction_prompt = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided documents as references (some of which might be irrelevant). Always cite at least one document for every sentences in the answer. Use the citation format of square brackets to indicate the cited documents (e.g., [1] for the first reference). If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. Only generate the answer, excluding any disclaimers, notes or list of references at the end of the answer."

instruction_prompt_mds = "Instruction: write 3 to 5 long statements that support the given document below. Each statements shouldbe standalone and has necessary context to understand the statements. All the statements together should cover the document as much as possible. Every statements should start with the format of square bracket with number (e.g., [1])."

instruction_prompt_mds = "Instruction: write 3 to 5 comprehensive statements that support the given document below. Each statement should be standalone and provide enough context to be understood independently. These statements should cover the document as thoroughly as possible. Each statement should begin with a number in square brackets (e.g., [1])."

instruction_prompt_doc = "Instruction: Break down the given document into 3 to 5 passages. Each passage should be clear, self-contained, and offer comprehensive context. Collectively, these statements should cover the main pooints of the document. Each statements should start with the format of number in square brackets (e.g., [1])."

instruction_prompt_q = "Instruction: Write a query based on the information of the given document below. The query can be either a question or a topic title as long as it covers the overall information of the document. The query should be standalone, understandable and comprehensive. Write only one query and within `<query>` and `</query>` tags."
# instruction_prompt_q = "Instruction: Based on the main information provided in the given document below, generate a single, clear, and standalone question. Ensure that the question is understandable on its own and that the answer can be directly found within the document. Write only one question."

demo_sep = "\n\n"
# doc_prompt_template = "Document [{ID}] {P}\n"
# demo_prompt_template = "{INST}\n\nDocuments:\n{D}\nStatements:\n{S}"
inst_prompt_template_c = "{INST}\n\n{DEMO}Document: {D}\n\nStatements:\n[1] {S}"
inst_prompt_template_q = "{INST}\n\nDocument: {D}\n\n{Q_PREFIX} {Q}"

def apply_docs_prompt(doc_items, ndoc=None, field='text'):
    p = ""
    for idx, doc_item in enumerate(doc_items[:ndoc]):
        p_doc = doc_prompt_template
        p_doc = p_doc.replace("{ID}", str(idx+1))
        p_doc = p_doc.replace("{T}", doc_item['title'])
        p_doc = p_doc.replace("{P}", doc_item[field])
        p += p_doc
    return p

def apply_demo_prompt(Q, D, A, instruction=""):
    p = demo_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D).replace("{A}", A)
    return p

def apply_inst_prompt_claim_gen(Q, D, instruction="", add_prefix=True):
    """ handle {DEMO} during the training. """
    p = inst_prompt_template_c
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{D}", D).replace("{S}", "")
    if add_prefix is False:
        p = p.replace("Statements:", "").strip()
    return p

def apply_inst_prompt_question_gen(Q, D, instruction="", prefix="<query>"):
    """ handle {DEMO} during the training. """
    p = inst_prompt_template_q
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{D}", D).replace("{Q}", "")
    p = p.replace("{Q_PREFIX}", prefix).strip()
    return p
