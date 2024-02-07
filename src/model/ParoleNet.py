import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
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
from src.metrics import *

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
    ):

        super(ParoleNet, self).__init__()

        self.dir_path = dir_path
        self.config = config
        self.model_dir = dir_path + f"/checkpoints/{config['name']}"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        config_parolenet = config["ParoleNet"]
        self.input_dim = config_parolenet["input_dim"]
        self.hidden_dim_1_layer = config_parolenet["hidden_dim_1_layer"]
        self.hidden_dim_2_layer = config_parolenet["hidden_dim_2_layer"]
        self.output_dim = config_parolenet["output_dim"]
        self.dropout_p = config_parolenet["dropout_p"]

        if self.input_dim == "None" or self.input_dim is None:
            self.linear1 = nn.LazyLinear(self.hidden_dim_1_layer)
        else:
            self.linear1 = nn.Linear(self.input_dim, self.hidden_dim_1_layer)
        self.linear2 = nn.Linear(self.hidden_dim_1_layer, self.hidden_dim_2_layer)
        self.linear3 = nn.Linear(self.hidden_dim_2_layer, self.output_dim)
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.logger = cu.create_logger(
            f"{config['name']}", self.model_dir + "/logs.txt"
        )

    ###################

    def parameters_number(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / (
            1000 * 1000
        )

    def forward_linear(self, input_data: torch.Tensor) -> torch.Tensor:

        linear1_output = torch.relu(self.linear1(input_data))
        linear2_output = torch.relu(self.linear2(self.dropout(linear1_output)))
        final_output = torch.softmax(self.linear3(self.dropout(linear2_output)), dim=1)

        return final_output

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:

        sample_tensor = torch.mean(input_data["audio"], dim=1)
        wave2vec_output = wave2vec_model(sample_tensor).last_hidden_state

        bert_output = bert_model(input_data["text"]).last_hidden_state

        combined_output = torch.cat(
            (wave2vec_output.flatten(start_dim=1), bert_output.flatten(start_dim=1)),
            dim=1,
        )

        final_output = self.forward_linear(combined_output)

        return final_output

    def evaluate(self, dataloader: DataLoader) -> float:

        self.eval()

        output = torch.tensor([]).to(DEVICE)
        labels = torch.tensor([]).to(DEVICE)

        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader, desc="Validating")):

                batch = cu.dict_to_device(batch, DEVICE)
                self.to(DEVICE)

                output = torch.cat((output, self.forward(batch)))
                labels = torch.cat((labels, batch["label"].long()))

        f1_0, precision_0, recall_0, _, _, _ = f1_score(output, labels, 0)
        f1_1, precision_1, recall_1, _, _, _ = f1_score(output, labels, 1)
        self.logger.info(
            f"Classe 0 | Precision: {precision_0 * 100:.2f}%, Recall: {recall_0 * 100:.2f}%, F1 Score: {f1_0 * 100:.2f}%"
        )
        self.logger.info(
            f"Classe 1 | Precision: {precision_1 * 100:.2f}%, Recall: {recall_1 * 100:.2f}%, F1 Score: {f1_1 * 100:.2f}%"
        )

        f1 = weighted_f1_score(output, labels)
        self.logger.info(f"Score : {f1}")

    def train_one_epoch(self, dataloader: DataLoader, optimizer) -> float:

        self.train(True)

        for _, batch in enumerate(tqdm(dataloader, desc="Training")):

            batch = cu.dict_to_device(batch, DEVICE)
            self.to(DEVICE)

            optimizer.zero_grad()

            output = self.forward(batch)
            labels = batch["label"].long()

            f1 = weighted_proba_f1_score(output, labels)

            epsilon = 1e-7
            loss = -torch.log(f1 + epsilon)

            loss.backward()

            optimizer.step()

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
            self.train_one_epoch(train_loader, optimizer)
            scheduler.step()

            # Validate on the validation subset
            self.logger.info(f"Validation :")
            self.evaluate(val_loader)
            self.logger.info(f"Test :")
            self.evaluate(test_loader)

    def launch(self, data: pd.core.frame.DataFrame, audio_path: str):
        try:
            self.train_loop(data, audio_path)
            self.logger.info(
                f"Nombre de paramètres du model: {self.parameters_number():.2f} M"
            )
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

    # Create the model
    model = ParoleNet(dir_path, config)

    # Test of initialization
    generator = DataGenerator(
        raw_data[:512], audio_path, **config["get_audio"], **config["get_text"]
    )
    dataloader = create_dataloader(generator, config["training"]["batch_size"])
    model.evaluate(dataloader)
    model.logger.info(
        f"Nombre de paramètres du model: {model.parameters_number():.2f} M"
    )
