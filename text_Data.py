import pandas as pd
from transformers import AutoTokenizer, DistilBertModel

file_path = 'data/transcr/AAOR_merge.csv'
data = pd.read_csv(file_path)

turn_column = data['turn_at_start']
text_column = data['text']


model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Phrase dont vous souhaitez obtenir les embeddings
phrase = text_column[10]
print(phrase)

# Tokenisation
inputs = tokenizer(phrase, return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.size())
