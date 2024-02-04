import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
sys.path.append(up(up(up(os.path.abspath(__file__)))))
import utils.custom_utils as cu
from src.dataloader.data_processing import *
from src.model.ParoleNet import *

if __name__ == "__main__":

    dir_path = up(up(os.path.abspath(__file__)))
    audio_path = dir_path + "/dataset/audio/2_channels/"
    raw_data = load_all_ipus(dir_path + "/dataset/transcr")

    raw_data = raw_data[:]

    # Load configuration
    config = cu.load_yaml_parameters(dir_path + "/config/config.yaml")
    config["ParoleNet"]["input_dim"] = get_input_dim(config)

    # Create the model
    model = ParoleNet(dir_path, config, **config["ParoleNet"])
    
    # Train
    model.launch(raw_data, audio_path)