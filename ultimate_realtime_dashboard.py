import streamlit as st
import torch
import numpy as np
import sounddevice as sd
import plotly.graph_objects as go
import time
from collections import deque
from models.seformer import SEFormer

st.set_page_config(layout="wide", page_title="Emotion AI Lab")

st.title("🎤 Emotion AI Laboratory - Live Dashboard")
st.markdown("### Real-Time Sliding Window Emotion Detection")

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

emotion_history = deque(maxlen=30)
confidence_history = deque(maxlen=30)

placeholder_graph = st.empty()
placeholder_gauge = st.empty()

run = st.checkbox("Start Live Monitoring")

if run:
    st.success("🎙 Live Emotion Monitoring Started")

    while run:
        audio = sd.rec(window_size, samplerate=fs, channels=1)
        sd.wait()

        audio = np.squeeze(audio)
        audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(audio_tensor)
            probs = torch.softmax(output, dim=1)[0]
            predicted = torch.argmax(probs).item()
            confidence = probs[predicted].item()

        emotion_history.append(predicted)
        confidence_history.append(confidence)

        # Timeline Graph
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

        placeholder_graph.plotly_chart(fig, use_container_width=True)

        # Confidence Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': f"Current Emotion: {emotion_map[predicted]}"},
            gauge={'axis': {'range': [0, 100]}}
        ))

        gauge.update_layout(template="plotly_dark", height=350)

        placeholder_gauge.plotly_chart(gauge, use_container_width=True)

        time.sleep(0.5)