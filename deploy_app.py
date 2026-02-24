import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
from collections import deque
from models.seformer import SEFormer

# ------------------ PAGE CONFIG ------------------
st.set_page_config(layout="wide", page_title="Emotion AI")
st.title("🎤 Real-Time Speech Emotion AI (Deployed Version)")

# ------------------ EMOTION MAP ------------------
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

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ LOAD MODEL (CACHED) ------------------
@st.cache_resource
def load_model():
    model = SEFormer()
    model.load_state_dict(torch.load("best_seformer.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# ------------------ SESSION STATE ------------------
if "model" not in st.session_state:
    st.session_state.model = None

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=20)

# ------------------ BUTTON TO LOAD MODEL ------------------
if st.button("🚀 Start Emotion Detection"):
    with st.spinner("Loading model..."):
        st.session_state.model = load_model()
    st.success("Model Loaded Successfully!")

# ------------------ AUDIO SETTINGS ------------------
fs = 16000
window_size = 4 * fs


# ------------------ AUDIO PROCESSOR ------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = np.zeros(0)

    def recv(self, frame):
        audio = frame.to_ndarray().flatten()
        self.buffer = np.concatenate([self.buffer, audio])

        if len(self.buffer) >= window_size and st.session_state.model is not None:
            chunk = self.buffer[-window_size:]
            audio_tensor = torch.tensor(chunk).unsqueeze(0).to(device)

            with torch.no_grad():
                output = st.session_state.model(audio_tensor)
                probs = torch.softmax(output, dim=1)[0]
                predicted = torch.argmax(probs).item()

            st.session_state.emotion_history.append(predicted)

        return frame


# ------------------ WEBRTC STREAM ------------------
webrtc_streamer(
    key="emotion-stream",
    mode="sendonly",
    audio_processor_factory=AudioProcessor,
)

# ------------------ PLOT ------------------
if len(st.session_state.emotion_history) > 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=list(st.session_state.emotion_history),
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

    current_emotion = emotion_map[st.session_state.emotion_history[-1]]
    st.success(f"Current Emotion: {current_emotion}")