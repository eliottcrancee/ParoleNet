
import torch
from tqdm import tqdm
from colorama import Fore, Style
from data_processing import *
from transformers import Wav2Vec2Model
from transformers import CamembertModel
from icecream import ic

########################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wave2vec_name = "facebook/wav2vec2-base-960h"
wave2vec_model = Wav2Vec2Model.from_pretrained(wave2vec_name)
wave2vec_model = wave2vec_model.to(device)

bert_name = 'camembert-base'
bert_model = CamembertModel.from_pretrained(bert_name)
bert_model = bert_model.to(device)

for param in wave2vec_model.parameters():
    param.requires_grad = False

for param in bert_model.parameters():
    param.requires_grad = False
    
########################################################################

class Model(torch.nn.Module):
    """
    Model using covolutional neural net architecture.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        
        super(Model, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define model components
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.1)
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
        
        wave2vec_output0 = torch.nn.functional.max_pool1d(wave2vec_output0, kernel_size=6)
        wave2vec_output1 = torch.nn.functional.max_pool1d(wave2vec_output1, kernel_size=6)
        
        bert_output = bert_model(input_data['text'])
        bert_output = bert_output.last_hidden_state
        
        wave2vec_output0 = torch.flatten(wave2vec_output0, start_dim=1)
        wave2vec_output1 = torch.flatten(wave2vec_output1, start_dim=1)
        bert_output = torch.flatten(bert_output, start_dim=1)

        # Concatenate or combine the outputs as needed
        combined_output = torch.cat((wave2vec_output0, wave2vec_output1, bert_output), dim=1)

        # Apply linear layers
        linear1_output = torch.relu(self.linear1(combined_output))
        final_output = self.linear2(self.dropout(linear1_output))

        return torch.softmax(final_output, dim=1)
    
    def evaluate(self, dataloader):
        
        self.to(self.device)
        
        self.eval()

        # Define the loss function
        loss_function = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        true_positive_1 = torch.tensor(0).to(self.device)
        false_positive_1 = torch.tensor(0).to(self.device)
        false_negative_1 = torch.tensor(0).to(self.device)
        
        true_positive_0 = torch.tensor(0).to(self.device)
        false_positive_0 = torch.tensor(0).to(self.device)
        false_negative_0 = torch.tensor(0).to(self.device)

        with torch.no_grad():  # Disable gradient computation during validation
            for _, batch in enumerate(tqdm(dataloader, desc="Validating")):
                
                output = self.forward(batch)
                labels = batch["label"].long()

                loss = loss_function(output, labels)
                total_loss += loss.item()

                _, predicted = torch.max(output, 1)
                
                true_positive_1 += torch.sum((predicted == labels) * (labels == 1))
                false_positive_1 += torch.sum((predicted == (1 - labels)) * ((1 - labels) == 1))
                false_negative_1 += torch.sum(((1-predicted) == labels) * (labels == 1))
                
                true_positive_0 += torch.sum((predicted == labels) * (labels == 0))
                false_positive_0 += torch.sum((predicted == (1 - labels)) * ((1 - labels) == 0))
                false_negative_0 += torch.sum(((1-predicted) == labels) * (labels == 0))
                
        precision_1 = true_positive_1 / max((true_positive_1 + false_positive_1), 1)
        recall_1 = true_positive_1 / max((true_positive_1 + false_negative_1), 1) 
        
        precision_0 = true_positive_0 / max((true_positive_0 + false_positive_0), 1)
        recall_0 = true_positive_0 / max((true_positive_0 + false_negative_0), 1) 
        
        f1_1 = 2 * (precision_1 * recall_1) / max((precision_1 + recall_1), 1)
        f1_0 = 2 * (precision_0 * recall_0) / max((precision_0 + recall_0), 1)
        
        print(f"Classe {Fore.RED}0{Style.RESET_ALL} | Precision: {Fore.GREEN}{precision_0 * 100:.2f}%{Style.RESET_ALL}, Recall: {Fore.GREEN}{recall_0 * 100:.2f}%{Style.RESET_ALL}, F1 Score: {Fore.GREEN}{f1_0 * 100:.2f}%{Style.RESET_ALL}")
        print(f"Classe {Fore.RED}1{Style.RESET_ALL} | Precision: {Fore.GREEN}{precision_1 * 100:.2f}%{Style.RESET_ALL}, Recall: {Fore.GREEN}{recall_1 * 100:.2f}%{Style.RESET_ALL}, F1 Score: {Fore.GREEN}{f1_1 * 100:.2f}%{Style.RESET_ALL}")
        print(f"Score : {(f1_0*0.18 + f1_1*(1-0.18))}")
                        
    def train_one_epoch(self, dataloader):
        
        self.train(True)
        
        optimizer = torch.optim.Adam(self.parameters())
        # loss_function = torch.nn.CrossEntropyLoss()
        
        try:
            for _, batch in enumerate(tqdm(dataloader, desc="Training")):
                
                try:
                
                    optimizer.zero_grad()
                    
                    output = self.forward(batch)
                    labels = batch["label"].long()
                    _, predicted = torch.max(output, 1)
                    
                    true_positive_0 = (output * (1 - labels).unsqueeze(1))[:, 0].sum()
                    false_positive_0 = ((1 - output) * labels.unsqueeze(1))[:, 0].sum()
                    false_negative_0 = ((1 - output) * (1 - labels).unsqueeze(1))[:, 0].sum()

                    true_positive_1 = (output * labels.unsqueeze(1))[:, 1].sum()
                    false_positive_1 = (output * (1 - labels).unsqueeze(1))[:, 1].sum()
                    false_negative_1 = ((1 - output) * labels.unsqueeze(1))[:, 1].sum()

                    epsilon = torch.tensor([1e-7]).to(self.device) # 1e-7
                    
                    precision_0 = true_positive_0 / (true_positive_0 + false_positive_0 + epsilon)
                    recall_0 = true_positive_0 / (true_positive_0 + false_negative_0 + epsilon)
                    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + epsilon)

                    precision_1 = true_positive_1 / (true_positive_1 + false_positive_1 + epsilon)
                    recall_1 = true_positive_1 / (true_positive_1 + false_negative_1 + epsilon)
                    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + epsilon)
                
                    loss = -torch.log((f1_0*0.18 + f1_1*(1-0.18)) + epsilon)
                    
                    # ic(output, labels)
                    # ic(true_positive_0, false_positive_0, false_negative_0, true_positive_1, false_positive_1, false_negative_1)
                    # ic(precision_0, recall_0, f1_0, precision_1, recall_1, f1_1)
                    # print("Successfully trained a batch")
                    
                    ic(loss)
                    loss.backward()
                    
                    # labels = batch["label"].long()
                    # loss = loss_function(output, labels)
                    # loss.backward()
                    
                    optimizer.step()
                    
                except StopIteration as e:
                    print(f"An error occurred: {e}")   
                    # Catch StopIteration and continue to the next batch
                    pass
        except Exception as e:
            # Handle other exceptions if needed
            print(f"An error occurred: {e}")      

    def train_loop(self, generator, nb_epoch):
        
        self.to(self.device)

        data = generator.raw_data

        train_data = data.sample(frac=0.9,random_state=200)
        test_data = data.drop(train_data.index)

        test_data.reset_index(drop=True, inplace=True)
        train_data.reset_index(drop=True, inplace=True)

        test_generator = DataGenerator(test_data, filepath)
        test_loader = create_dataloader(test_generator)
        
        for epoch_number in range(nb_epoch):

            train_subdata = train_data.sample(frac=0.9,random_state=200)
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

generator = DataGenerator(raw_data.iloc[8192:8192+4096], filepath)

# model = Model(27904, 16, 2, device)
# dataloader = create_dataloader(generator)
# model.evaluate(dataloader)

# print("Nombre de paramètres du model:", model.parameters_number())

# model.train_loop(generator, 5)