import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import librosa
from models.seformer import SEFormer

# Emotion labels
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

# Recording settings
duration = 4  # seconds
fs = 16000    # sampling rate

print("Recording will start in 2 seconds...")
sd.sleep(2000)

print("🎤 Speak now...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()

# Save recording
audio = np.squeeze(audio)
write("test.wav", fs, audio)

print("Recording Complete ✅")

# Load audio using librosa
audio, sr = librosa.load("test.wav", sr=16000)

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

print("🎯 Predicted Emotion:", emotion_map[predicted])