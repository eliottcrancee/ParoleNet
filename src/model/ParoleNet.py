import torch
from tqdm import tqdm
from transformers import Wav2Vec2Model
from transformers import CamembertModel
from icecream import ic

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
sys.path.append(up(up(up(os.path.abspath(__file__)))))
import utils.custom_utils as cu
from src.dataloader.data_processing import *

########################################################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WAVE2VEC_NAME = "facebook/wav2vec2-base-960h"
wave2vec_model = Wav2Vec2Model.from_pretrained(WAVE2VEC_NAME)
wave2vec_model = wave2vec_model.to(DEVICE)

BERT_NAME = "camembert-base"
bert_model = CamembertModel.from_pretrained(BERT_NAME)
bert_model = bert_model.to(DEVICE)

for param in wave2vec_model.parameters():
    param.requires_grad = False

for param in bert_model.parameters():
    param.requires_grad = False

########################################################################


class ParoleNet(torch.nn.Module):
    """
    Model using dense neural architecture.
    """

    def __init__(
        self,
        dir_path: str,
        config: dict,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_p: float,
    ):

        super(ParoleNet, self).__init__()

        self.dir_path = dir_path
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim*4)
        self.linear2 = torch.nn.Linear(hidden_dim*4, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)

        self.config = config
        self.model_dir = dir_path + f"/checkpoints/{config['name']}"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.logger = cu.create_logger(
            f"{config['name']}", self.model_dir + "/logs.txt"
        )
        self.logger.info(f"Nombre de paramètres du model: {self.parameters_number()}")

    ###################

    def parameters_number(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:

        sample_tensor = torch.mean(input_data["audio"], dim=1)
        wave2vec_output = wave2vec_model(sample_tensor).last_hidden_state

        bert_output = bert_model(input_data["text"]).last_hidden_state

        combined_output = torch.cat(
            (wave2vec_output.flatten(start_dim=1), bert_output.flatten(start_dim=1)),
            dim=1,
        )

        linear1_output = torch.relu(self.linear1(combined_output))
        linear2_output = torch.relu(self.linear2(self.dropout(linear1_output)))
        final_output = torch.softmax(self.linear3(self.dropout(linear2_output)), dim=1)

        return final_output

    def evaluate(self, dataloader: DataLoader, turn_after: float) -> float:

        self.eval()

        true_positive_1 = torch.tensor(0).to(DEVICE)
        false_positive_1 = torch.tensor(0).to(DEVICE)
        false_negative_1 = torch.tensor(0).to(DEVICE)

        true_positive_0 = torch.tensor(0).to(DEVICE)
        false_positive_0 = torch.tensor(0).to(DEVICE)
        false_negative_0 = torch.tensor(0).to(DEVICE)
        n_0 = torch.tensor(0).to(DEVICE)
        n_1 = torch.tensor(0).to(DEVICE)

        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader, desc="Validating")):

                batch = {key: value.to(DEVICE) for key, value in batch.items()}
                self.to(DEVICE)

                output = self.forward(batch)
                labels = batch["label"].long()

                _, predicted = torch.max(output, 1)

                true_positive_1 += torch.sum((predicted == labels) * (labels == 1))
                false_positive_1 += torch.sum(
                    (predicted == (1 - labels)) * ((1 - labels) == 1)
                )
                false_negative_1 += torch.sum(
                    ((1 - predicted) == labels) * (labels == 1)
                )

                true_positive_0 += torch.sum((predicted == labels) * (labels == 0))
                false_positive_0 += torch.sum(
                    (predicted == (1 - labels)) * ((1 - labels) == 0)
                )
                false_negative_0 += torch.sum(
                    ((1 - predicted) == labels) * (labels == 0)
                )

                self.logger.debug(predicted, labels)
                self.logger.debug(
                    true_positive_0,
                    false_positive_0,
                    false_negative_0,
                    true_positive_1,
                    false_positive_1,
                    false_negative_1,
                )
                
                n_0 += torch.sum(labels == 0)
                n_1 += torch.sum(labels == 1)

        epsilon = 1e-10
        precision_1 = true_positive_1 / (true_positive_1 + false_positive_1 + epsilon)
        recall_1 = true_positive_1 / (true_positive_1 + false_negative_1 + epsilon)

        precision_0 = true_positive_0 / (true_positive_0 + false_positive_0 + epsilon)
        recall_0 = true_positive_0 / (true_positive_0 + false_negative_0 + epsilon)

        f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + epsilon)
        f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + epsilon)

        # ic(precision_0, recall_0, f1_0, precision_1, recall_1, f1_1)

        self.logger.info(
            f"Classe 0 | Precision: {precision_0 * 100:.2f}%, Recall: {recall_0 * 100:.2f}%, F1 Score: {f1_0 * 100:.2f}%"
        )
        self.logger.info(
            f"Classe 1 | Precision: {precision_1 * 100:.2f}%, Recall: {recall_1 * 100:.2f}%, F1 Score: {f1_1 * 100:.2f}%"
        )
        
        w_0 = (n_0 + n_1) / (2 * n_0 + epsilon)
        w_1 = (n_0 + n_1) / (2 * n_1 + epsilon)

        f1_weighted = (w_0 * f1_0 + w_1 * f1_1) / (w_0 + w_1)

        self.logger.info(f"Score : {f1_weighted}")

    def train_one_epoch(
        self, dataloader: DataLoader, optimizer, turn_after: float
    ) -> float:

        self.train(True)
        
        try:
            for _, batch in enumerate(tqdm(dataloader, desc="Training")):
                try:

                    batch = {key: value.to(DEVICE) for key, value in batch.items()}
                    self.to(DEVICE)

                    optimizer.zero_grad()

                    output = self.forward(batch)
                    labels = batch["label"].long()
                    
                    epsilon = torch.tensor([1e-10]).to(DEVICE)

                    true_positive_0 = (output * (1 - labels).unsqueeze(1))[:, 0].sum()
                    false_positive_0 = ((1 - output) * labels.unsqueeze(1))[:, 0].sum()
                    false_negative_0 = ((1 - output) * (1 - labels).unsqueeze(1))[
                        :, 0
                    ].sum()

                    true_positive_1 = (output * labels.unsqueeze(1))[:, 1].sum()
                    false_positive_1 = (output * (1 - labels).unsqueeze(1))[:, 1].sum()
                    false_negative_1 = ((1 - output) * labels.unsqueeze(1))[:, 1].sum()

                    precision_0 = true_positive_0 / (
                        true_positive_0 + false_positive_0 + epsilon
                    )
                    recall_0 = true_positive_0 / (
                        true_positive_0 + false_negative_0 + epsilon
                    )
                    f1_0 = (
                        2
                        * (precision_0 * recall_0)
                        / (precision_0 + recall_0 + epsilon)
                    )

                    precision_1 = true_positive_1 / (
                        true_positive_1 + false_positive_1 + epsilon
                    )
                    recall_1 = true_positive_1 / (
                        true_positive_1 + false_negative_1 + epsilon
                    )
                    f1_1 = (
                        2
                        * (precision_1 * recall_1)
                        / (precision_1 + recall_1 + epsilon)
                    )
                    
                    n_0 = torch.sum(labels == 0)
                    n_1 = torch.sum(labels == 1)
                    
                    w_0 = (n_0 + n_1) / (2 * n_0 + epsilon)
                    w_1 = (n_0 + n_1) / (2 * n_1+ epsilon)

                    f1_weighted = (w_0 * f1_0 + w_1 * f1_1) / (w_0 + w_1)
                    
                    loss = -torch.log(f1_weighted + epsilon)

                    # ic(output, labels)
                    # ic(true_positive_0, false_positive_0, false_negative_0, true_positive_1, false_positive_1, false_negative_1)
                    # ic(precision_0, recall_0, f1_0, precision_1, recall_1, f1_1)
                    # ic(loss)
                    # print("Successfully trained a batch")

                    loss.backward()

                    optimizer.step()

                except StopIteration as e:
                    self.logger.error(f"An error occurred: {e}")
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")

    def train_loop(self, data: pd.core.frame.DataFrame, audio_path: str):

        self.to(DEVICE)

        test_data = data.sample(
            frac=self.config["sampling"]["test_fraction"],
            random_state=self.config["sampling"]["seed"],
        )
        train_data = data.drop(test_data.index)

        test_data.reset_index(drop=True, inplace=True)
        train_data.reset_index(drop=True, inplace=True)

        test_generator = DataGenerator(
            test_data, audio_path, **self.config["get_audio"], **self.config["get_text"]
        )
        test_loader = create_dataloader(
            test_generator, self.config["training"]["batch_size"]
        )

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config["training"]["scheduler"]["initial_lr"]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config["training"]["scheduler"]["step_size"],
            gamma=self.config["training"]["scheduler"]["gamma"],
        )

        for epoch_number in range(self.config["training"]["nb_epochs"]):

            val_data = train_data.sample(
                frac=self.config["sampling"]["val_fraction"],
                random_state=self.config["sampling"]["seed"],
            )
            train_subdata = train_data.drop(val_data.index)

            train_subdata.reset_index(drop=True, inplace=True)
            val_data.reset_index(drop=True, inplace=True)

            train_generator = DataGenerator(
                train_subdata,
                audio_path,
                **self.config["get_audio"],
                **self.config["get_text"],
            )
            val_generator = DataGenerator(
                val_data,
                audio_path,
                **self.config["get_audio"],
                **self.config["get_text"],
            )

            train_loader = create_dataloader(
                train_generator, self.config["training"]["batch_size"]
            )
            val_loader = create_dataloader(
                val_generator, self.config["training"]["batch_size"]
            )

            self.logger.info(f"EPOCH {epoch_number + 1}:")

            # Train for one epoch
            self.train_one_epoch(train_loader, optimizer, **self.config["evaluate"])
            scheduler.step()

            # Validate on the validation subset
            self.logger.info(f"Validation :")
            self.evaluate(val_loader, **self.config["evaluate"])
            self.logger.info(f"Test :")
            self.evaluate(test_loader, **self.config["evaluate"])

    def launch(self, data: pd.core.frame.DataFrame, audio_path: str):
        try:
            self.train_loop(data, audio_path)
            self.save_model()
        except Exception as e:
            print(f"An error occurred: {e}")
            self.save_model()

    def save_model(self):
        torch.save(self, self.model_dir + f"/{self.config['name']}.pt")
        cu.save_yaml_parameters(self.config, self.model_dir + f"/config.yaml")


########################################################################

if __name__ == "__main__":

    dir_path = up(up(up(os.path.abspath(__file__))))
    audio_path = dir_path + "/dataset/audio/2_channels/"
    raw_data = load_all_ipus(dir_path + "/dataset/transcr")

    # Load configuration
    config = cu.load_yaml_parameters(dir_path + "/config/config.yaml")
    config["ParoleNet"]["input_dim"] = get_input_dim(config)

    # Create the model
    model = ParoleNet(dir_path, config, **config["ParoleNet"])

    # Test of initialization
    generator = DataGenerator(
        raw_data[:512], audio_path, **config["get_audio"], **config["get_text"]
    )
    dataloader = create_dataloader(generator, config["training"]["batch_size"])
    model.evaluate(dataloader, **config["evaluate"])
