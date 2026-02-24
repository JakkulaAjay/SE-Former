import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
from collections import deque
from models.seformer import SEFormer

st.set_page_config(layout="wide", page_title="Emotion AI")

st.title("🎤 Real-Time Speech Emotion AI (Deployed Version)")

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

fs = 16000
window_size = 4 * fs

emotion_history = deque(maxlen=20)

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = np.zeros(0)

    def recv(self, frame):
        audio = frame.to_ndarray().flatten()
        self.buffer = np.concatenate([self.buffer, audio])

        if len(self.buffer) >= window_size:
            chunk = self.buffer[-window_size:]
            audio_tensor = torch.tensor(chunk).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(audio_tensor)
                probs = torch.softmax(output, dim=1)[0]
                predicted = torch.argmax(probs).item()

            emotion_history.append(predicted)

        return frame

webrtc_streamer(
    key="emotion-stream",
    mode="sendonly",
    audio_processor_factory=AudioProcessor,
)

if len(emotion_history) > 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=list(emotion_history),
        mode='lines+markers'
    ))

    fig.update_layout(
        title="Live Emotion Timeline",
        yaxis=dict(
            tickmode='array',
            tickvals=list(emotion_map.keys()),
            ticktext=list(emotion_map.values())
        ),
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(f"Current Emotion: {emotion_map[emotion_history[-1]]}")