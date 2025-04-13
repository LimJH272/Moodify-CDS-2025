import streamlit as st
import numpy as np
import torch
from torch import nn
import torchaudio
import librosa
import io
import os

# Import from your project files
from models import NilsHMeierCNN
from data_preparation import (
    Stereo2Mono,
    NormaliseMelSpec,
    TARGET_SAMPLE_RATE,
    TARGET_AUDIO_LENGTH,
    INT_TO_LABEL
)

# --- Model Loading ---


@st.cache_resource  # Cache the model loading
def load_pretrained_model(model_path='mood_model.pth', feature='melspecs'):
    """Loads the pretrained NilsHMeierCNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the model structure (ensure feature matches training)
    model = NilsHMeierCNN(feature=feature)
    # Load the saved state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Set the model to evaluation mode
    model.eval()
    # Move model to the appropriate device
    model.to(device)
    return model, device

# --- Preprocessing Function ---


def preprocess_audio(audio_bytes, target_sr=TARGET_SAMPLE_RATE, target_len=TARGET_AUDIO_LENGTH, device='cpu'):
    """Preprocesses raw audio bytes to the model's expected input format."""
    buffer = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(buffer)

    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        # Check if Stereo2Mono is needed or if simple averaging works
        # Using simple mean for now, adjust if Stereo2Mono class is essential and available
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # If Stereo2Mono class is strictly needed:
        # mono_converter = Stereo2Mono()
        # waveform = mono_converter(waveform).unsqueeze(0) # Add channel dim back if needed

    # Pad or truncate to target length
    current_len = waveform.shape[1]
    if current_len < target_len:
        padding = target_len - current_len
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif current_len > target_len:
        waveform = waveform[:, :target_len]

    # --- Feature Extraction (Mel Spectrogram) ---
    n_mels = 128
    n_fft = 4096  # Match parameters from data_preparation.py
    melspec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_mels=n_mels,
        n_fft=n_fft,
        center=True,
    ).to(device)

    melspec = melspec_transform(waveform.to(device))

    # --- Normalization ---
    # NormaliseMelSpec expects numpy, uses librosa.power_to_db
    # Convert to numpy, apply normalization, convert back to tensor
    # Remove channel dim for librosa
    melspec_np = melspec.squeeze(0).cpu().numpy()

    # Use librosa power_to_db for normalization like in NormaliseMelSpec
    # Ensure ref=np.max behaves correctly for single spectrograms
    melspec_db = librosa.power_to_db(melspec_np, ref=np.max(melspec_np))

    melspec_tensor = torch.from_numpy(melspec_db).unsqueeze(
        0).to(device)  # Add batch dimension back

    # Model expects a dictionary {'melspecs': tensor}
    # The model's forward adds the channel dimension
    return {'melspecs': melspec_tensor}


# --- Label Mapping ---
def get_mood_label(class_idx):
    """Maps class index to mood label using INT_TO_LABEL from data_preparation."""
    return INT_TO_LABEL.get(class_idx, "Unknown")  # Use .get for safety


# --- Streamlit App ---
st.title("Moodify Music Classifier")

# File upload section
uploaded_files = st.file_uploader(
    "Upload audio files",
    type=["wav", "mp3"],
    accept_multiple_files=True  # Enable multiple files
)

# Load model only once
try:
    model, device = load_pretrained_model()  # Use the cached function
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop execution if model fails to load


run_button = st.button("Analyze Mood")


if run_button and uploaded_files:  # Check if any files are uploaded
    results = []

    for uploaded_file in uploaded_files:  # Process each file
        with st.spinner(f'Analyzing {uploaded_file.name}...'):
            try:
                audio_bytes = uploaded_file.read()
                input_features = preprocess_audio(audio_bytes, device=device)

                with torch.no_grad():
                    outputs = model(input_features)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_idx = probabilities.argmax().item()
                    predicted_label = get_mood_label(predicted_idx)

                results.append({
                    "filename": uploaded_file.name,
                    "label": predicted_label,
                    "probabilities": probabilities.squeeze().cpu().numpy()
                })

            except Exception as e:
                results.append({
                    "filename": uploaded_file.name,
                    "error": str(e)
                })

    # Display all results
    st.subheader("Analysis Results")

    for result in results:
        with st.expander(f"{result['filename']}"):
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.write(f"**Predicted Mood:** {result['label']}")
                st.write("**Confidence Scores:**")
                for i, prob in enumerate(result['probabilities']):
                    st.write(f"{get_mood_label(i)}: {prob:.4f}")

elif run_button:
    st.warning("Please upload audio files first.")
