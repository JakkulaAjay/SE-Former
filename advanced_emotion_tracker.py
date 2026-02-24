import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import librosa
from collections import Counter
import matplotlib.pyplot as plt
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
duration = 20   # seconds
fs = 16000
chunk_duration = 4  # seconds
stride_duration = 2 # overlap stride

print("Recording will start in 2 seconds...")
sd.sleep(2000)

print("🎤 Speak continuously for 20 seconds...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()

audio = np.squeeze(audio)
write("advanced_test.wav", fs, audio)

print("Recording Complete ✅")

# Load audio
audio, sr = librosa.load("advanced_test.wav", sr=16000)

chunk_size = fs * chunk_duration
stride = fs * stride_duration

predictions = []
confidences = []
time_stamps = []

# Sliding window processing
for start in range(0, len(audio) - chunk_size, stride):
    chunk = audio[start:start + chunk_size]
    chunk_tensor = torch.tensor(chunk).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(chunk_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item()

    predictions.append(pred)
    confidences.append(confidence)
    time_stamps.append(start / fs)

# Print segment results
print("\n🔎 Segment-wise Predictions:")
for i in range(len(predictions)):
    print(f"Time {time_stamps[i]:.1f}s - {time_stamps[i]+chunk_duration:.1f}s → "
          f"{emotion_map[predictions[i]]} "
          f"({confidences[i]*100:.2f}%)")

# Majority vote
final_prediction = Counter(predictions).most_common(1)[0][0]

print("\n🎯 Final Predicted Emotion:",
      emotion_map[final_prediction])

# Plot emotion timeline
emotion_numeric = predictions

plt.figure()
plt.plot(time_stamps, emotion_numeric)
plt.yticks(list(emotion_map.keys()), list(emotion_map.values()))
plt.xlabel("Time (seconds)")
plt.ylabel("Emotion")
plt.title("Emotion Timeline Over Speech")
plt.show()