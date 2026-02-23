import torch
import librosa
import os
from models.seformer import SEFormer

# Emotion mapping
emotion_map = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fear",
    6: "Disgust",
    7: "Surprised"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = SEFormer()
model.load_state_dict(torch.load("best_seformer.pth", map_location=device))
model.to(device)
model.eval()

print("Model Loaded Successfully ✅")

# Ask user for audio path
audio_path = input("Enter audio file path: ")

if not os.path.exists(audio_path):
    print("File not found ❌")
    exit()

# Load audio
audio, sr = librosa.load(audio_path, sr=16000)

# Pad or trim to 4 seconds
max_length = 16000 * 4
if len(audio) < max_length:
    padding = max_length - len(audio)
    audio = torch.nn.functional.pad(torch.tensor(audio), (0, padding))
else:
    audio = torch.tensor(audio[:max_length])

audio = audio.unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(audio)
    predicted = torch.argmax(output, dim=1).item()

print("🎤 Predicted Emotion:", emotion_map[predicted])