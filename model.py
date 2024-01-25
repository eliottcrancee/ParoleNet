
import torch
from tqdm import tqdm
from colorama import Fore, Style
from data_processing import *
from transformers import Wav2Vec2Model
from transformers import CamembertModel
import numpy as np

########################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wave2vec_name = "facebook/wav2vec2-base-960h"
wave2vec_model = Wav2Vec2Model.from_pretrained(wave2vec_name)
wave2vec_model = wave2vec_model.to(device)

bert_name = 'camembert-base'
bert_model = CamembertModel.from_pretrained(bert_name)
bert_model = bert_model.to(device)

print(sum(p.numel() for p in bert_model.parameters()))

for param in wave2vec_model.parameters():
    param.requires_grad = False

for param in bert_model.parameters():
    param.requires_grad = False

########################################################################

class F1Loss(torch.nn.Module):

    def forward(self, predicted, labels):
        
        print(predicted, labels)
        
        true_positive = torch.logical_and((predicted == 1), (labels == 1)).sum().item()
        false_positive = torch.logical_and((predicted == 1), (labels == 0)).sum().item()
        false_negative = torch.logical_and((predicted == 0), (labels == 1)).sum().item()
        
        precision = true_positive / max((true_positive + false_positive), 1)
        recall = true_positive / max((true_positive + false_negative), 1)
        
        f1_score = torch.tensor(2 * (precision * recall) / max((precision + recall), 1))
        
        loss = -torch.log(f1_score)

        return loss
    
########################################################################

class Model(torch.nn.Module):
    """
    Model using covolutional neural net architecture.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define model components
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        
###################
        
    def parameters_number(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
        
    def forward(self, input_data):
        # Define the forward pass using the model components
        
        input_data['audio'] = input_data['audio'].to(self.device)
        input_data['text'] = input_data['text'].to(self.device)
        input_data['label'] = input_data['label'].to(self.device)
        
        
        channel0 = input_data['audio'][:,0,:]
        channel1 = input_data['audio'][:,1,:]
        wave2vec_output0 = wave2vec_model(channel0)
        wave2vec_output1 = wave2vec_model(channel1)
        wave2vec_output0 = wave2vec_output0.last_hidden_state
        wave2vec_output1 = wave2vec_output1.last_hidden_state
        bert_output = bert_model(input_data['text'])
        bert_output = bert_output.last_hidden_state
        
        wave2vec_output0 = torch.flatten(wave2vec_output0, start_dim=1)
        wave2vec_output1 = torch.flatten(wave2vec_output1, start_dim=1)
        bert_output = torch.flatten(bert_output, start_dim=1)

        # Concatenate or combine the outputs as needed
        combined_output = torch.cat((wave2vec_output0, wave2vec_output1, bert_output), dim=1)

        # Apply linear layers
        linear1_output = torch.relu(self.linear1(combined_output))
        final_output = self.linear2(linear1_output)

        return torch.sigmoid(final_output)
    
    def evaluate(self, dataloader):
        self.eval()

        # Define the loss function
        loss_function = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        true_positive = 0.0
        false_positive = 0.0
        false_negative = 0.0

        with torch.no_grad():  # Disable gradient computation during validation
            for _, batch in enumerate(tqdm(dataloader, desc="Validating")):
                
                output = self.forward(batch)
                labels = batch["label"].long()

                loss = loss_function(output, labels)
                total_loss += loss.item()
                
                print(output, labels)

                _, predicted = torch.max(output, 1)
                
                true_positive += torch.logical_and((predicted == 1), (labels == 1)).sum().item()
                false_positive += torch.logical_and((predicted == 1), (labels == 0)).sum().item()
                false_negative += torch.logical_and((predicted == 0), (labels == 1)).sum().item()
                
        precision = true_positive / max((true_positive + false_positive), 1)
        recall = true_positive / max((true_positive + false_negative), 1)
        
        f1_score = 2 * (precision * recall) / max((precision + recall), 1)
        
        print(f"Precision: {Fore.GREEN}{precision * 100:.2f}%{Style.RESET_ALL}, Recall: {Fore.GREEN}{recall * 100:.2f}%{Style.RESET_ALL}, F1 Score: {Fore.GREEN}{recall * 100:.2f}%{Style.RESET_ALL}")
        
        return precision, recall, f1_score
                        
    def train_one_epoch(self, dataloader):
        
        self.train(True)
        
        optimizer = torch.optim.Adam(self.parameters())
        loss_function = torch.nn.CrossEntropyLoss()
        # loss_function = F1Loss()

        for _, batch in enumerate(tqdm(dataloader, desc="Training")):
            
            optimizer.zero_grad()
            
            output = self.forward(batch)
            labels = batch["label"].long()
            
            # _, predicted = torch.max(output, 1)
            # loss = loss_function(predicted, labels)
            
            labels = batch["label"].long()
            loss = loss_function(output, labels)

            loss.backward()
            optimizer.step()

        return None         

    def train_loop(self, generator, nb_epoch):
        
        self.to(self.device)

        data = generator.raw_data

        train_data = data.sample(frac=0.8,random_state=200)
        test_data = data.drop(train_data.index)

        test_data.reset_index(drop=True, inplace=True)
        train_data.reset_index(drop=True, inplace=True)

        test_generator = DataGenerator(test_data, filepath)
        test_loader = create_dataloader(test_generator)
        
        self.evaluate(test_loader)
        
        for epoch_number in range(nb_epoch):

            train_subdata = train_data.sample(frac=0.8,random_state=200)
            val_subdata = train_data.drop(train_subdata.index)
            
            train_subdata.reset_index(drop=True, inplace=True)
            val_subdata.reset_index(drop=True, inplace=True)
            
            train_subgenerator = DataGenerator(train_subdata, filepath)
            val_subgenerator = DataGenerator(val_subdata, filepath)

            train_loader = create_dataloader(train_subgenerator)
            val_loader = create_dataloader(val_subgenerator)

            print("")    
            print(f'{Fore.GREEN}EPOCH {epoch_number + 1}:{Style.RESET_ALL}')
            
            # Train for one epoch
            self.train_one_epoch(train_loader)

            # Validate on the validation subset
            print(f'{Fore.CYAN}Validation :{Style.RESET_ALL}')
            self.evaluate(val_loader)
            print(f'{Fore.YELLOW}Test :{Style.RESET_ALL}')
            self.evaluate(test_loader)

########################################################################

generator = DataGenerator(raw_data.iloc[:400], filepath)
model = Model(90624, 16, 2)

print("Nombre de paramètres du model:", model.parameters_number())

model.train_loop(generator, 1)