import torch
import torchaudio
from load_data import *
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torch.utils.data import Dataset

data = load_all_ipus("Dataset/transcr")

model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

def inference_wav2vec(n): 
    
    audio_file_path = "Dataset/audio/2_channels/" + data["dyad"][n].replace("transcr\\","") + ".wav"
    
    audio_tensor, frame_rate = torchaudio.load(audio_file_path)

    stop = int(frame_rate*data["stop"][n])
    
    start = int(frame_rate*(data["stop"][n]-2))
     
    sample_tensor = audio_tensor[:,start:stop]
    
    print(sample_tensor.shape)
    
    result = model.forward(sample_tensor).last_hidden_state
    
    return result

def get_audio(i):
    audio_file_path = "Dataset/audio/2_channels/" + data["dyad"][i].replace("transcr\\","") + ".wav"
    audio_tensor, frame_rate = torchaudio.load(audio_file_path)
    stop = int(frame_rate*data["stop"][i])
    start = int(frame_rate*(data["stop"][i]-2))
    sample_tensor = audio_tensor[:,start:stop]
    print(sample_tensor.shape)
    return sample_tensor.flatten(end_dim=1)

def get_text

class donn(Dataset):
    
    def __getitem__(i):
        return {"audio" : get_audio(i),
                "text" : get_text(i),
                "label" : get_label(i)}