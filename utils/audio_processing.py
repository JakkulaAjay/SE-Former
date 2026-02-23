# utils/audio_processing.py

import librosa
import torch
import numpy as np
from config import SAMPLE_RATE, N_MELS

def extract_features(file_path):
    """
    Loads audio file and extracts log-mel spectrogram features.
    Returns tensor of shape (1, n_mels, time_steps)
    """

    # Load audio
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Extract mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS
    )

    # Convert to log scale
    log_mel = librosa.power_to_db(mel)

    # Fix time dimension to 128
    log_mel = librosa.util.fix_length(log_mel, size=128, axis=1)

    # Convert to tensor
    log_mel = torch.tensor(log_mel).unsqueeze(0).float()

    return log_mel