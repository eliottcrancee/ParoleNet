import torchaudio
import wave
from load_data import *
from transformers import Wav2Vec2Model, Wav2Vec2Processor

mp4_file_path = "Dataset/audio/2_channels/AA_OR.wav"

data = load_all_ipus("Dataset/transcr")

model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

def get_embeddings(n, mp4_file_path) : 
    audio_tensor, frame_rate = torchaudio.load(mp4_file_path)
    start = int(frame_rate*data["start"][n])
    stop = int(frame_rate*data["stop"][n])
    sample_tensor = audio_tensor[:,start:stop]
    result = model.forward(sample_tensor).last_hidden_state
    return(result)