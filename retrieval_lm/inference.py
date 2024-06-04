#!/usr/bin/env python
# coding=utf-8

##### [NOTE] vllm only suport GPU compatibility greater than 7

MODEL='TinyLlama/TinyLlama-1.1B-Chat-v0.6'
# MODEL=meta-llama/Llama-2-7b-hf

from vllm.vllm import LLM, SamplingParams

model = LLM(MODEL, dtype="half")
sampling_params = SamplingParams(
        temperature=0.0, 
        top_p=1.0, 
        max_tokens=100, 
        skip_special_tokens=False
)

def format_prompt(input, paragraph=None):
  prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
  if paragraph is not None:
    prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
  return prompt

query_1 = "Leave odd one out: twitter, instagram, whatsapp."
query_2 = "Can you tell me the difference between llamas and alpacas?"
queries = [query_1, query_2]

# for a query that doesn't require retrieval
preds = model.generate(
        [format_prompt(query) for query in queries], sampling_params
)
for pred in preds:
  print("Model prediction: {0}".format(pred.outputs[0].text))

