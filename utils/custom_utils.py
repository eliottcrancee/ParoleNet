import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

import yaml
import logging
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from itertools import product


def get_tensor_memory_size(tensor: torch.Tensor) -> int:
    """
    Calculate the memory size of a given tensor.
    """
    return tensor.element_size() * tensor.numel()


def load_yaml_parameters(file_path: str) -> dict:
    """
    Load YAML parameters from a file.
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_yaml_parameters(config: dict, file_path: str):
    """
    Save parameters to a YAML file.
    """
    with open(file_path, "w") as f:
        yaml.dump(config, f)


def create_logger(name: str, filename: str, level=logging.INFO) -> logging.Logger:
    """
    Create a logger with the specified name and output file.
    """
    logging.basicConfig(filename=filename)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def dict_to_device(dict, device):
    return {key: value.to(device) for key, value in dict.items()}


def xgboost_find_best_hyperparameters(filename):

    results = pd.read_csv(filename, sep=";")
    y = results["score"].values
    X = results[
        [
            "dropout_p",
            "hidden_dim_1_layer",
            "hidden_dim_2_layer",
            "initial_lr",
            "gamma",
            "batch_size",
            "step_size",
        ]
    ].values

    xgb_model = XGBRegressor(objective="reg:squarederror")
    xgb_model.fit(X, y)

    hyperparameters_to_evaluate = {
        "dropout_p": np.linspace(0.1, 1, 5),
        "hidden_dim_1_layer": np.linspace(32, 256, 8),
        "hidden_dim_2_layer": np.linspace(16, 128, 8),
        "initial_lr": np.linspace(0.0001, 0.01, 5),
        "gamma": np.linspace(0.01, 0.2, 5),
        "batch_size": [32, 64, 96, 128],
        "step_size": [2, 4, 6, 8],
    }

    hyperparameter_combinations = list(product(*hyperparameters_to_evaluate.values()))

    # Predict scores for each hyperparameter combination
    predicted_scores = []
    for _, hyperparams in enumerate(tqdm(hyperparameter_combinations, desc="Learning")):
        score = xgb_model.predict([hyperparams])[0]
        predicted_scores.append(score)

    # Find hyperparameters with highest predicted score
    best_hyperparameters = hyperparameter_combinations[
        predicted_scores.index(max(predicted_scores))
    ]
    print(
        "Best hyperparameters:",
        dict(zip(hyperparameters_to_evaluate.keys(), best_hyperparameters)),
    )
    print("Predicted score:", max(predicted_scores))
