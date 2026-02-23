import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class SEFormer(nn.Module):
    def __init__(self, num_classes=8, freeze_layers=9):
        """
        num_classes: number of emotion classes
        freeze_layers: number of initial transformer layers to freeze
        """
        super(SEFormer, self).__init__()

        # Load pretrained Wav2Vec2 base model
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        # -------------------------------
        # 1️⃣ Freeze CNN Feature Extractor
        # -------------------------------
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False

        # -------------------------------
        # 2️⃣ Freeze first N transformer layers
        # -------------------------------
        for layer in self.wav2vec.encoder.layers[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        # -------------------------------
        # 3️⃣ Classification Head
        # -------------------------------
        hidden_size = self.wav2vec.config.hidden_size  # 768

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_values):
        """
        input_values: (batch_size, sequence_length)
        """

        outputs = self.wav2vec(input_values)
        hidden_states = outputs.last_hidden_state
        # shape: (batch_size, time_steps, 768)

        # Mean pooling across time dimension
        pooled = torch.mean(hidden_states, dim=1)

        x = self.dropout(pooled)
        logits = self.classifier(x)

        return logits