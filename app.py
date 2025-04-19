from data_preparation import (
    Stereo2Mono,
    NormaliseMelSpec,
    TARGET_SAMPLE_RATE,
    TARGET_AUDIO_LENGTH,
    INT_TO_LABEL
)
from models import ImprovedEmotionTransformer, NilsHMeierCNN, VGGStyleCNN
import streamlit as st
import numpy as np
import torch
from torch import nn
import torchaudio
import librosa
import io
import os

os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER"] = "false"

WEIGHTS = {
    'cnn': 0.373,
    'transformer': 0.527,
    'vgg': 0.100
}


@st.cache_resource
def load_ensemble_models():
    """Load both models silently"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Nils CNN
    cnn = NilsHMeierCNN(feature='melspecs',out_features=4)
    cnn.load_state_dict(torch.load('mood_model.pth', map_location=device))
    cnn.eval().to(device)

    # Load Transformer
    transformer = ImprovedEmotionTransformer()
    transformer.load_state_dict(torch.load(
        'balanced2.pth', map_location=device))
    transformer.eval().to(device)

    # Load VGG model
    vgg = VGGStyleCNN(feature='melspecs')
    vgg.load_state_dict(torch.load('best_jig.pt', map_location=device))
    vgg.eval().to(device)

    return cnn, transformer, vgg, device


def ensemble_predict(models, features, device):
    """Combine predictions from all models"""
    cnn, transformer, vgg = models
    with torch.no_grad():
        cnn_out = torch.softmax(cnn(features), dim=1)
        trans_out = torch.softmax(transformer(features), dim=1)
        vgg_out = torch.softmax(vgg(features), dim=1)

        probs = (
            WEIGHTS['cnn'] * cnn_out +
            WEIGHTS['transformer'] * trans_out +
            WEIGHTS['vgg'] * vgg_out
        )
    return probs

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


# Initialize session state for tracking resets
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

# Function to trigger reset


def clear_uploaded_files():
    # Increment counter to force rerun
    st.session_state.reset_counter += 1
    # Clear any results (optional)
    if 'results' in st.session_state:
        st.session_state.results = []
    # This forces the file uploader to reset
    st.session_state.file_uploader = []


# --- Streamlit App ---
st.title("Moodify Music Classifier")

# Load model only once
try:
    cnn_model, transformer_model, vgg_model, device = load_ensemble_models()  # Add vgg_model
    st.success("3-Model system loaded successfully")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# File upload section
uploaded_files = st.file_uploader(
    "Upload audio files",
    type=["wav", "mp3"],
    accept_multiple_files=True,
    # Key changes on reset
    key=f"file_uploader_{st.session_state.reset_counter}"
)


col1, col2 = st.columns([5, 1])
with col1:
    run_button = st.button("Analyze Mood")
with col2:
    clear_button = st.button("Clear Files", on_click=clear_uploaded_files)


if run_button and uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        with st.spinner(f'Analyzing {uploaded_file.name}...'):
            try:
                audio_bytes = uploaded_file.read()
                input_features = preprocess_audio(audio_bytes, device=device)

                # Get ensemble prediction
                combined_probs = ensemble_predict(
                    (cnn_model, transformer_model, vgg_model),
                    input_features,
                    device
                )

                # Rest of processing remains the same
                predicted_idx = combined_probs.argmax().item()
                predicted_label = get_mood_label(predicted_idx)

                results.append({
                    "filename": uploaded_file.name,
                    "label": predicted_label,
                    "probabilities": combined_probs.squeeze().cpu().numpy()
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
