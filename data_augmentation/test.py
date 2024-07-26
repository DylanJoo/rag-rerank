# from align_pseudo_entailment import compute_claims 
#
# a = "A person on a horse jumps over a broken down airplane."
# b = ["A person is at a diner, ordering an omelette.", "A person is outdoors, on a horse"]
#
# c = compute_claims(a, b)
# print(c)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model = AutoModelForSeq2SeqLM.from_pretrained(
    'google/t5_xxl_true_nli_mixture', 
    torch_dtype=torch.bfloat16, 
)
tokenizer = AutoTokenizer.from_pretrained(
    'google/t5_xxl_true_nli_mixture'
)

# a = "At least three people are injured after a man firebombs a fast-food restaurant in the Cologne main station and takes a woman hostage. The woman is later rescued by German counter-terrorism units."
# b = "A DeLand woman was rescued from her abusive acquaintance after passing a note to staff at an animal hospital, asking them to call authorities"
a = "At least three people are injured after a man firebombs a fast-food restaurant in the Cologne main station and takes a woman hostage. The woman is later rescued by German counter-terrorism units."
b = "The number of people affected by the stabbing was not limited to just one person."
input_text = f"premise: {a} hypothesis: {b}"


input = tokenizer(input_text, return_tensors='pt')
outputs = model.generate(**input)
outputs = model.generate(
    **input,
    max_new_tokens=10,
    return_dict_in_generate=True, 
    output_scores=True
)

print(outputs.sequences)
o = outputs.scores[0][0].log_softmax(-1)
o2 = outputs.scores[1][0].log_softmax(-1)

true = (o[209] + o2[1])
false = (o[3] + o2[632])
print(true, false)
# score = torch.stack([false, true]).softmax(-1).detach().numpy().tolist()[1]
# print((o[209] + o2[1]).exp())
# print(o[3] + o2[632])

input_text = "premise: hello world. hypothesis: leave me alone."
input = tokenizer(input_text, return_tensors='pt')
outputs = model.generate(**input)
outputs = model.generate(
    **input,
    max_new_tokens=10,
    return_dict_in_generate=True, 
    output_scores=True
)

print(outputs.sequences)
o = outputs.scores[0][0].log_softmax(-1)
o2 = outputs.scores[1][0].log_softmax(-1)

true = (o[209] + o2[1])
false = (o[3] + o2[632])
print(true, false)

# score = torch.stack([false, true]).softmax(-1).detach().numpy().tolist()[1]
# print((o[209] + o2[1]).exp())
# print(o[3] + o2[632])
