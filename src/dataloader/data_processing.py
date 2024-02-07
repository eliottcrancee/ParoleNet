import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import Wav2Vec2Processor
from transformers import CamembertTokenizer

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
sys.path.append(up(up(up(os.path.abspath(__file__)))))
import utils.custom_utils as cu
from src.dataloader.load_data import *

########################################################################

WAVE2VEC_NAME = "facebook/wav2vec2-base-960h"
audio_processor = Wav2Vec2Processor.from_pretrained(WAVE2VEC_NAME)

BERT_NAME = "camembert-base"
text_tokenizer = CamembertTokenizer.from_pretrained(BERT_NAME)

########################################################################


def get_audio(
    i: int, raw_data: pd.core.frame.DataFrame, audio_path: str, duration: float = 1
) -> torch.Tensor:
    """
    Load audio data from file and return a tensor sample.
    """
    audio_file_path = audio_path + raw_data["dyad"][i].replace("transcr\\", "") + ".wav"
    sampling_rate = torchaudio.info(audio_file_path).sample_rate
    stop = int(sampling_rate * raw_data["stop"][i])
    start = stop - int(sampling_rate * duration)

    if start < 0:
        sample_tensor, _ = torchaudio.load(audio_file_path, 0, stop)
        sample_tensor = torch.mean(sample_tensor, dim=0)
        sample_tensor = audio_processor(
            sample_tensor, return_tensors="pt", sampling_rate=sampling_rate
        ).input_values.squeeze(0)
        sample_tensor = torch.cat((torch.zeros(abs(start)), sample_tensor))
    else:
        sample_tensor, _ = torchaudio.load(audio_file_path, start, stop - start)
        sample_tensor = torch.mean(sample_tensor, dim=0)
        sample_tensor = audio_processor(
            sample_tensor, return_tensors="pt", sampling_rate=sampling_rate
        ).input_values.squeeze(0)
    return sample_tensor


def get_text(
    i: int,
    raw_data: pd.core.frame.DataFrame,
    context_length: int = 20,
    pad_token: str = "<pad>",
) -> torch.Tensor:
    """
    Tokenizes the text data and returns the tensor.
    """
    pad_token_id = text_tokenizer(pad_token)["input_ids"][1]
    text = raw_data["text"][i]
    if type(text) == str:
        text_tokenized = text_tokenizer(text, return_tensors="pt")["input_ids"]
    else:
        text_tokenized = torch.tensor([[pad_token_id] * context_length])
    text_tokenized = torch.cat(
        (torch.tensor([[pad_token_id] * context_length]), text_tokenized), dim=1
    )
    text_tokenized = text_tokenized[:, -context_length:]
    return text_tokenized.squeeze(0)


def get_label(i: int, raw_data: pd.core.frame.DataFrame) -> float:
    """
    Get the label for the given index from the raw data.
    """
    return raw_data["turn_after"].astype("float32")[i]


def get_input_dim(config: dict) -> int:
    """
    Get the size of the input layer for the dense neural layer.
    """
    audio_size = int(50 * config["get_audio"]["duration"] - 0.5)
    text_size = config["get_text"]["context_length"]
    return (audio_size + text_size) * 768


class DataGenerator(Dataset):

    def __init__(
        self,
        raw_data: pd.core.frame.DataFrame,
        audio_path: str,
        duration: float,
        context_length: int,
        pad_token: str,
    ) -> None:
        self.raw_data = raw_data
        self.audio_path = audio_path
        self.duration = duration
        self.context_length = context_length
        self.pad_token = pad_token

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "audio": get_audio(i, self.raw_data, self.audio_path, self.duration),
            "text": get_text(i, self.raw_data, self.context_length, self.pad_token),
            "label": get_label(i, self.raw_data),
        }

    def __len__(self) -> int:
        return len(self.raw_data)


def create_dataloader(generator: DataGenerator, batch_size: int = 32) -> DataLoader:

    dataloader = DataLoader(
        generator,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return dataloader


########################################################################

if __name__ == "__main__":

    dir_path = up(up(up(os.path.abspath(__file__))))
    print(dir_path)
    audio_path = dir_path + "/dataset/audio/2_channels/"
    raw_data = load_all_ipus(dir_path + "/dataset/transcr")

    print("Text shape: ", get_text(0, raw_data).shape)
    print("Audio shape: ", get_audio(0, raw_data, audio_path, 1).shape)

    # generator = DataGenerator(raw_data, audio_path)
    # dataloader = create_dataloader(generator)
