import os
import torch
import librosa
from torch.utils.data import Dataset

class RAVDESSDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.labels = []
        
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".wav"):
                    emotion = int(file.split("-")[2]) - 1
                    self.files.append(os.path.join(subdir, file))
                    self.labels.append(emotion)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        audio, sr = librosa.load(file_path, sr=16000)

        max_length = 16000 * 4  # 4 seconds
        if len(audio) < max_length:
            padding = max_length - len(audio)
            audio = torch.nn.functional.pad(torch.tensor(audio), (0, padding))
        else:
            audio = torch.tensor(audio[:max_length])

        return audio.float(), torch.tensor(label).long()