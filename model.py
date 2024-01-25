
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from colorama import Fore, Style

########################################################################

class Model(torch.nn.Module):
    """
    Model using covolutional neural net architecture.
    """
    def __init__(self, s):

        super(GRU, self).__init__()

        # Extracting parameters from the provided object
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = parameters.output_dim
        self.hidden_dim = parameters.hidden_dim
        self.n_layers = parameters.n_layers
        self.drop_prob = parameters.drop_prob
        self.is_bidirectional = parameters.is_bidirectional

        # Define model components
        self.emb = torch.nn.Embedding(parameters.output_dim, parameters.embedding_dim)
        self.gru = torch.nn.GRU(
            parameters.embedding_dim,
            parameters.hidden_dim,
            parameters.n_layers,
            batch_first=True,
            bidirectional=self.is_bidirectional,
            dropout=parameters.drop_prob if parameters.n_layers > 1 else 0
        )
        self.do = torch.nn.Dropout(p=parameters.drop_prob)
        self.fc = torch.nn.Linear((2 if self.is_bidirectional else 1)*parameters.hidden_dim, parameters.output_dim)
        self.relu = torch.nn.ReLU()
        
###################
        
    def parameters_number(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
        
    def forward(self, x):
        """
        Defines the forward pass of the GRU model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Hidden state tensor.
        """
        # Apply embedding layer to input
        embedded = self.emb(x)

        # Apply GRU layer
        out, h = self.gru(embedded)

        # Apply dropout, ReLU activation, and linear layer
        out = self.fc(self.relu(self.do(out)))

        return out, h
    
    def predict(self, word):
        """
        Predicts the output word given an input word using the trained GRU model.

        Args:
            word (str): Input word for prediction.
            letter_dict_path (str): Path to the JSON file containing the letter dictionary.

        Returns:
            str: Predicted word.
        """
        # Load letter dictionary from JSON file
        with open(self.letter_dict_path, 'r', encoding='utf-8') as file:
            letter_dict = json.load(file)

        # Get padding value from letter dictionary
        pad = letter_dict.get("#", 0)

        # Convert input word to tensor using letter dictionary
        letters_w = torch.tensor([letter_dict.get(i, pad) for i in word] + [pad] * 6)
        letters_w = letters_w.to(self.device)

        # Forward pass through the GRU model
        out, h = self.forward(letters_w)

        # Find the index with the highest probability in the output tensor
        out = torch.argmax(out, dim=1)

        # Convert indices back to letters using the letter dictionary
        letter = list(letter_dict.keys())
        predicted_word = "".join([letter[int(index)] for index in out])
        return predicted_word.replace("#","")
    
    def validate(self, dataloader):
        """
        Validates the GRU model on the provided dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for validation.

        Returns:
            float: Average validation loss.
            float: Validation accuracy.
        """
        # Set the model to evaluation mode
        self.eval()

        # Define the loss function
        loss_function = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct_predictions = 0
        flechies = 0
        correct_predictions_flechies = 0
        total_samples = 0

        with torch.no_grad():  # Disable gradient computation during validation
            for _, batch in enumerate(tqdm(dataloader, desc="Validating")):
                batch = batch.to(self.device)
                x, y = batch[:,0], batch[:,1]

                # Forward pass
                Y_prime, h = self.forward(x)

                for i, example in enumerate(batch):
                    x, y = example[0].to(self.device), example[1].to(self.device)
                    y_prime = Y_prime[i]

                    # Ensure input and target sizes match
                    if x.size(0) == y.size(0):

                        # Calculate loss
                        loss = loss_function(y_prime, y)
                        total_loss += loss.item()

                        # Calculate accuracy
                        predicted_labels = torch.argmax(y_prime, dim=1)
                        if torch.equal(predicted_labels, y) :
                            correct_predictions += torch.sum(predicted_labels == y).item()
                        
                        if not torch.equal(x, y) :
                            if torch.equal(predicted_labels, y):
                                correct_predictions_flechies += torch.sum(predicted_labels == y).item()
                        
                            flechies += y.size(0)

                        total_samples += y.size(0)

        # Calculate average loss and accuracy
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        accuracy_flechies = correct_predictions_flechies / flechies
        print("Pourcentage de flechies :", flechies / total_samples)

        # Print validation results
        print(f"Loss: {Fore.RED}{avg_loss:.4f}{Style.RESET_ALL}, Accuracy: {Fore.GREEN}{accuracy * 100:.2f}%{Style.RESET_ALL}, Accuracy Flechies: {Fore.BLUE}{accuracy_flechies * 100:.2f}%{Style.RESET_ALL}")

        return avg_loss, accuracy
    
    def train_one_epoch(self, dataloader):
        """
        Trains the GRU model for one epoch on the provided dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for training.
        """
        # Set the model to training mode
        self.train(True)

        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters())
        loss_function = torch.nn.CrossEntropyLoss()

        # Iterate over batches in the dataloader
        for i, batch in enumerate(tqdm(dataloader, desc="Training")):
            batch = batch.to(self.device)
            x, y = batch[:,0], batch[:,1]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            y_prime, h = self.forward(x)

            y = torch.flatten(y, end_dim=1)
            y_prime = torch.flatten(y_prime, end_dim=1)

            # Calculate loss for the entire batch
            loss = loss_function(y_prime, y)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            

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