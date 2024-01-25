from transformers import AutoTokenizer, DistilBertModel
import torch
import numpy as np

# Charger le modèle et le tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Phrase dont vous souhaitez obtenir les embeddings
phrase = "<pad> <pad>"

# Tokenisation
inputs = tokenizer(phrase, return_tensors="pt")
print(inputs)
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.size())
