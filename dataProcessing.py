import torch
from torch.utils.data import Dataset, DataLoader
from conllu import parse, parse_incr, TokenList
from torch.nn.utils.rnn import pad_sequence
from load_data import *


dataset = load_all_ipus("Dataset/transcr")
batch_size = 32
print(dataset["ipu_id"][1])

dataset_index = [(torch.tensor([k,dataset["turn_after"].astype(int)[k]])) for k in range(len(dataset))]

dataloader = DataLoader(dataset_index, batch_size=batch_size, shuffle=True)
print(dataloader.dataset)