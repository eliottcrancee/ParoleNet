import yaml
import torch
import logging


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

def create_logger(name: str, filename: str, level = logging.INFO) -> logging.Logger:
    """
    Create a logger with the specified name and output file.
    """
    logging.basicConfig(filename=filename)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
