import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class SEFormer(nn.Module):
    def __init__(self, num_classes=8):
        super(SEFormer, self).__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # Freeze feature extractor
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False

        # Freeze first 9 transformer layers
        for layer in self.wav2vec.encoder.layers[:9]:
            for param in layer.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        hidden_states = outputs.last_hidden_state  # (B, T, 768)

        pooled = torch.mean(hidden_states, dim=1)
        x = self.dropout(pooled)
        return self.fc(x)