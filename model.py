
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from colorama import Fore, Style
from dataProcessing import *

########################################################################

class Model(torch.nn.Module):
    """
    Model using covolutional neural net architecture.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, wave2vec_model, bert_model):
        
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define model components
        self.wave2vec_model = wave2vec_model
        self.bert_model = bert_model
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        
###################
        
    def parameters_number(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
        
    def forward(self, input_data):
        # Define the forward pass using the model components
        
        channel0 = input_data['audio'][:,0,:]
        channel1 = input_data['audio'][:,1,:]
        wave2vec_output0 = self.wave2vec_model(channel0)
        wave2vec_output1 = self.wave2vec_model(channel1)
        wave2vec_output0 = wave2vec_output0.last_hidden_state
        wave2vec_output1 = wave2vec_output1.last_hidden_state
        bert_output = self.bert_model(input_data['text'])
        bert_output = bert_output.last_hidden_state
        
        print(wave2vec_output0.shape, wave2vec_output1.shape, bert_output.shape)
        
        wave2vec_output0 = torch.flatten(wave2vec_output0, start_dim=1)
        wave2vec_output1 = torch.flatten(wave2vec_output1, start_dim=1)
        bert_output = torch.flatten(bert_output, start_dim=1)
        
        print(wave2vec_output0.shape, wave2vec_output1.shape, bert_output.shape)

        # Concatenate or combine the outputs as needed
        combined_output = torch.cat((wave2vec_output0, wave2vec_output1, bert_output), dim=1)
        
        print(combined_output.shape)

        # Apply linear layers
        linear1_output = torch.relu(self.linear1(combined_output))
        final_output = self.linear2(linear1_output)

        return final_output

    def train_one_epoch(self, dataloader):
        
        self.train(True)
        
        optimizer = torch.optim.Adam(self.parameters())
        loss_function = torch.nn.CrossEntropyLoss()

        for _, batch in enumerate(tqdm(dataloader, desc="Training")):
            
            optimizer.zero_grad()
            
            output = self.forward(batch)
            
            # output = output.long()
            labels = batch["label"].long()

            loss = loss_function(output, labels)

            loss.backward()
            optimizer.step()

        return None         

    def train_loop(self, dataset, nb_epoch):
        """
        Training loop for the GRU model.

        Args:
            dataset (torch.utils.data.Dataset): Dataset for training.
            nb_epoch (int): Number of training epochs.
        """
        # gru_model = GRU(parameters=gru_params)
        # gru_model.train_loop(dataset=your_dataset, nb_epoch=10)
        
        self.to(self.device)  # Move the model to the GPU if available

        data = dataset.data

        train_set, test_set = torch.utils.data.random_split(data, [int(len(data) * 0.95), len(data) - int(len(data) * 0.95)])
        test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=4096)

        for epoch_number in range(nb_epoch):

            # Split the dataset into training and validation subsets
            train_subset, val_subset = torch.utils.data.random_split(train_set, [int(len(train_set) * 0.95), len(train_set) - int(len(train_set) * 0.95)])

            # Create dataloaders for training and validation
            train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=4096)
            val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=4096)

            print("")    
            print(f'{Fore.GREEN}EPOCH {epoch_number + 1}:{Style.RESET_ALL}')
            
            # Train for one epoch
            self.train_one_epoch(train_loader)

            # Validate on the validation subset
            print(f'{Fore.CYAN}Validation :{Style.RESET_ALL}')
            self.validate(val_loader)
            print(f'{Fore.YELLOW}Test :{Style.RESET_ALL}')
            self.validate(test_loader)

    def save(self):
        filename = f"{self.embedding_dim}emb_{self.hidden_dim}hidden_{self.n_layers}layer"+("_bidirectional" if self.is_bidirectional else None)+".pth" 
        torch.save(self, filename)

########################################################################

dataloader = create_dataloader(raw_data, filepath)
model = Model(90624, 64, 2, wave2vec_model, bert_model)
model.train_one_epoch(dataloader)