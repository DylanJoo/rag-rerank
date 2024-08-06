# example_doc = "National Archives\nYes, it’s that time again, folks. It’s the first Friday of the month, when for one ever-so-brief moment the interests of Wall Street, Washington and Main Street are all aligned on one thing: Jobs. \n \n A fresh update on the U.S. employment situation for January hits the wires at 8:30 a.m. New York time offering one of the most important snapshots on how the economy fared during the previous month. Expectations are for 203,000 new jobs to be created, according to economists polled by Dow Jones Newswires, compared to 227,000 jobs added in February. The unemployment rate is expected to hold steady at 8.3%. \n \n Here at MarketBeat HQ, we’ll be offering color commentary before and after the data crosses the wires. Feel free to weigh-in yourself, via the comments section. And while you’re here, why don’t you sign up to follow us on Twitter. \n \n Enjoy the show."

###################################
# prompt for statement generation #
###################################
template_statement = "Instruction: {INST}\n\nDocument: {D}\n\n{PREFIX}"
instruction_statement = "Write 10 diverse statements that are evident from the given document. Each statement should be standalone and have necessary context. Every statements should start with the format of a square bracket with number (e.g., [1])."
def prompt_statement_gen(INST="", D="", PREFIX="Statements:\n[1]: "):
    p = template_statement
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{D}", D)
    p = p.replace("{PREFIX}", PREFIX).strip()
    return p

###################################
# prompt for statement generation #
###################################
# [this is for 2round generation] = "Write corrsponding questions for each of the given list of statements. The origianl context is also provided as reference. Each question should contain the necessary context if needed. Ensure the questions follow the same order of statements. Every questions should start with the format of a square bracket with number (e.g., [1]), like the statements."
template_question = "Instruction: {INST}\n\nDocument: {D}\n\n{PREFIX}"
instruction_question = "Write 10 diverse questions that can reveal the information contained in the given document. Each question should be self-contained and have necessary context. Write the question within `<q>` and `</q>` tags."
def prompt_question_gen(INST="", D="", PREFIX="Questions:\n<q>"):
    p = template_question
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{D}", D)
    p = p.replace("{PREFIX}", PREFIX).strip()
    return p

###################################
# prompt for summaries generation #
###################################
template_summary = "Instruction: {INST}\n\nDocument: {D}\n\n{PREFIX}"
# instruction_summary = "Break down the given document into 2 to 3 standalone summaries that have similar length. Each summaries should be a self-contained passage with necessary context. Keep the wording in summaries as similar as possible to the original document. Each summaries should start with the format of a square bracket with number (e.g., [1])."
instruction_summary = "Break down the given document into 2-3 standalone paragraphs of approximately 200 words each, providing essential context and information. Use similar wording and phrasing as the original document. Write each paragraph within `<p>` and `</p>` tags."
# instruction_summary = "Write two standalone summaries for the given document."
def prompt_summary_gen(INST="", D="", PREFIX="Paragraphs:\n<p>"):
    p = template_summary
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{D}", D)
    p = p.replace("{PREFIX}", PREFIX).strip()
    return p

################################
# prompt for rating generation #
################################
# guideline = "- 5: The answer is highly relevant, complete, and accurate.\n- 4: The answer is mostly relevant and complete but may have minor gaps or inaccuracies.\n- 3: The answer is partially relevant and complete, with noticeable gaps or inaccuracies.\n- 2: The answer has limited relevance and completeness, with significant gaps or inaccuracies.\n- 1: The answer is minimally relevant or complete, with substantial shortcomings.\n- 0: The answer is not relevant or complete at all."
guideline = "- 0: The context provides no relevant information.\n- 1: The context provides very little relevant information.\n- 2: The context provides some relevant information but is mostly insufficient.\n- 3: The context provides an adequate amount of relevant information.\n- 4: The context provides a good amount of relevant information.\n- 5: The context provides all necessary information in detail."
template_rating = "Instruction: {INST}\n\nGuideline:\n{G}\n\nQuestion: {Q}\n\nContext: {C}\n\n{PREFIX}" 
instruction_rating = "Determine whether the question can be answered based on the provided context? Rate the context with on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating."
def prompt_rating_gen(INST="", Q="", C="", PREFIX="Rating:"):
    p = template_rating
    p = p.replace("{G}", guideline)
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{Q}", Q)
    p = p.replace("{C}", C)
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

