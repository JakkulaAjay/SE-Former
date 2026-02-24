import streamlit as st
import torch
import numpy as np
import librosa
import plotly.graph_objs as go
import time
from collections import deque
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration
from models.seformer import SEFormer

st.set_page_config(page_title="Real-Time Emotion AI", layout="wide")

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

@st.cache_resource
def load_model():
    model = SEFormer()
    model.load_state_dict(torch.load("best_seformer.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

st.title("🎤 Real-Time Speech Emotion Intelligence Dashboard")
st.markdown("### Live Emotion Tracking with Sliding Window Inference")

emotion_history = deque(maxlen=30)
time_history = deque(maxlen=30)

fs = 16000
window_seconds = 10   # Increased duration
chunk_seconds = 4

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = np.zeros(0)

    def recv(self, frame):
        audio = frame.to_ndarray().flatten()
        self.buffer = np.concatenate([self.buffer, audio])

        if len(self.buffer) >= fs * window_seconds:
            audio_chunk = self.buffer[-fs * chunk_seconds:]
            audio_tensor = torch.tensor(audio_chunk).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(audio_tensor)
                probs = torch.softmax(output, dim=1)[0]
                predicted = torch.argmax(probs).item()

            emotion_history.append(predicted)
            time_history.append(time.time())

        return frame

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="emotion-stream",
    mode="sendonly",
    audio_processor_factory=AudioProcessor,
    rtc_configuration=RTC_CONFIGURATION,
)

# Live Plot
if len(emotion_history) > 0:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(emotion_history))),
        y=list(emotion_history),
        mode='lines+markers',
        line=dict(width=4),
    ))

    fig.update_layout(
        title="Live Emotion Timeline",
        xaxis_title="Time Window",
        yaxis=dict(
            tickmode='array',
            tickvals=list(emotion_map.keys()),
            ticktext=list(emotion_map.values())
        ),
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    current_emotion = emotion_map[emotion_history[-1]]
    st.success(f"🎯 Current Emotion: {current_emotion}")