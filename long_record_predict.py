import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import librosa
from collections import Counter
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

# Load model
model = SEFormer()
model.load_state_dict(torch.load("best_seformer.pth", map_location=device))
model.to(device)
model.eval()

print("Model Loaded Successfully ✅")

# Recording settings
duration = 15   # seconds (change if needed)
fs = 16000

print("Recording will start in 2 seconds...")
sd.sleep(2000)

print("🎤 Speak continuously for 15 seconds...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()

audio = np.squeeze(audio)
write("long_test.wav", fs, audio)

print("Recording Complete ✅")

# Load audio
audio, sr = librosa.load("long_test.wav", sr=16000)

chunk_size = 16000 * 4  # 4 seconds per chunk
predictions = []

# Split into chunks
for start in range(0, len(audio), chunk_size):
    chunk = audio[start:start + chunk_size]

    if len(chunk) < chunk_size:
        break

    chunk_tensor = torch.tensor(chunk).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(chunk_tensor)
        predicted = torch.argmax(output, dim=1).item()
        predictions.append(predicted)

# Majority vote
if len(predictions) > 0:
    final_prediction = Counter(predictions).most_common(1)[0][0]

    print("\nChunk Predictions:")
    for i, pred in enumerate(predictions):
        print(f"Segment {i+1}: {emotion_map[pred]}")

    print("\n🎯 Final Predicted Emotion (Majority Vote):",
          emotion_map[final_prediction])
else:
    print("Audio too short for processing.")