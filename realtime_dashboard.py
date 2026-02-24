import streamlit as st
import torch
import sounddevice as sd
import numpy as np
import librosa
import time
import matplotlib.pyplot as plt
from collections import deque
from models.seformer import SEFormer

st.set_page_config(layout="wide")

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

# Load model only once
@st.cache_resource
def load_model():
    model = SEFormer()
    model.load_state_dict(torch.load("best_seformer.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

st.title("🎤 Real-Time Speech Emotion Monitoring Dashboard")

# Session storage
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []
    st.session_state.time_history = []
    st.session_state.last_emotion = None

fs = 16000
duration = 4

# Record audio button
if st.button("🎙 Record 4 Seconds"):
    st.info("Recording... Speak Now!")

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    audio = np.squeeze(audio)
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(audio_tensor)
        probs = torch.softmax(output, dim=1)[0]
        predicted = torch.argmax(probs).item()
        confidence = probs[predicted].item()

    current_time = time.time()

    st.session_state.emotion_history.append(predicted)
    st.session_state.time_history.append(current_time)

    # Alert on emotion change
    if (
        st.session_state.last_emotion is not None
        and predicted != st.session_state.last_emotion
    ):
        st.warning(
            f"⚠ Emotion Changed: {emotion_map[st.session_state.last_emotion]} ➜ {emotion_map[predicted]}"
        )

    st.session_state.last_emotion = predicted

    st.success(
        f"Detected Emotion: {emotion_map[predicted]} ({confidence*100:.2f}%)"
    )

# Plot graph
if len(st.session_state.emotion_history) > 0:
    fig, ax = plt.subplots()
    ax.plot(
        range(len(st.session_state.emotion_history)),
        st.session_state.emotion_history,
        marker="o"
    )
    ax.set_yticks(list(emotion_map.keys()))
    ax.set_yticklabels(list(emotion_map.values()))
    ax.set_xlabel("Recording Segment")
    ax.set_ylabel("Emotion")
    ax.set_title("Emotion Trend Over Time")

    st.pyplot(fig)