from fidt5 import FiDT5
from transformers import AutoTokenizer

model = FiDT5.from_pretrained('google/flan-t5-large')
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')

model = FiDT5.from_pretrained('google/flan-t5-large')
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')

inputs = tokenizer([
    'question: what is xxx512? context: xxe512 is a dog.', 
    'question: what is xxx512? context: x3x512 is a cat.', 
    'question: what is xxx512? context: xxx512 is a car.', 
    'question: what is banana? context: banana is not a dog', 
    'question: what is banana? context: banana is not a fruit.', 
    'question: what is banana? context: banana is a car', 
], padding=True, truncation=True, return_tensors='pt')

input_ids = inputs['input_ids'].view(2, 3, -1)
attention_mask = inputs['attention_mask'].view(2, -1)


output = model.generate(input_ids=input_ids, attention_mask=attention_mask)
print(output)
print(tokenizer.decode(output[0]))
print(tokenizer.decode(output[1]))
