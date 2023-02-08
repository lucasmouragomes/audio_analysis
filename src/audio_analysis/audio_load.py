import numpy as np
import librosa
import warnings
from sys import path
from tqdm import tqdm

def load_audio(files, n_mels=90, duration=30, offset=0):
    n_mels = n_mels
    warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
    audio_data = []
    for i in tqdm(range(len(files.values))):
        f = files.values[i]
        audio_data.append(read_audio_data(f, n_mels=n_mels, duration=duration, offset=offset))
    print('Audio data succesfully processed!')
    return audio_data
    
def read_audio_data(f, n_mels, duration, offset):
    
    y, sr = librosa.load(f, offset=offset, duration=duration, dtype=None)
    y=y.astype(np.float32)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)[0]
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y, pad=False)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    result = {
        'chroma': chroma_stft,
        'rmse': rmse,
        'spec_cent': spec_cent,
        'spec_bw': spec_bw,
        'rolloff': rolloff,
        'zcr': zcr,
        'mfcc': mfcc,
        'melspectrogram': melspectrogram,
        'y': y
    }
    return result