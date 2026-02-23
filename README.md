# SE-Former: Hybrid CNN–Transformer for Speech Emotion Recognition

## Overview
This project implements a Speech Emotion Recognition (SER) system using a CNN baseline model.
Future work integrates Transformer-based models (Wav2Vec2) for contextual modeling.

## Dataset
RAVDESS Dataset
- 8 emotion classes
- Audio sampled at 16kHz
- Log-Mel Spectrogram features used

## Project Structure
- audio_processing.py → Feature extraction
- dataset.py → Custom dataset class
- train.py → Model training
- config.py → Hyperparameters

## How to Run

1. Install requirements:
pip install -r requirements.txt

2. Place dataset in:
data/raw_audio/

3. Run:
python train.py