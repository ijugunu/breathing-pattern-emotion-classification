import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from scipy.signal import find_peaks, butter, sosfiltfilt
from scipy.fft import fft, fftfreq

st.set_page_config(page_title="Breathing Emotion Classifier", layout="wide")

st.title("Breathing Pattern Emotion Classification")
st.markdown("Classifies emotions (Baseline / Stress / Amusement) from respiration signals using Machine Learning.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_names.txt") as f:
        feature_names = [line.strip() for line in f if line.strip()]
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

SOS = butter(4, [0.1/(700/2), 0.35/(700/2)], btype="band", output="sos")

def extract_features(signal, sr=700):
    signal = signal - np.mean(signal)
    filtered = sosfiltfilt(SOS, signal)

    feats = {
        "mean": np.mean(filtered),
        "std": np.std(filtered),
        "max": np.max(filtered),
        "min": np.min(filtered),
        "range": np.max(filtered) - np.min(filtered),
        "rms": np.sqrt(np.mean(filtered**2)),
        "slope": np.polyfit(range(len(filtered)), filtered, 1)[0]
    }

    peaks, _ = find_peaks(filtered, distance=sr*0.3)
    troughs, _ = find_peaks(-filtered, distance=sr*0.3)

    if len(peaks) > 1:
        intervals = np.diff(peaks) / sr
        feats["breathing_rate"] = 60 / np.mean(intervals)
        feats["breathing_rate_std"] = 60 * np.std(intervals)
    else:
        feats["breathing_rate"] = 0
        feats["breathing_rate_std"] = 0

    inhale_times, exhale_times, depths = [], [], []
    for i in range(min(len(peaks)-1, len(troughs))):
        if peaks[i] < troughs[i]:
            inhale_times.append((troughs[i] - peaks[i]) / sr)
            exhale_times.append((peaks[i+1] - troughs[i]) / sr)
            depths.append(filtered[peaks[i]] - filtered[troughs[i]])

    feats["avg_inhale_time"] = np.mean(inhale_times) if inhale_times else 0
    feats["avg_exhale_time"] = np.mean(exhale_times) if exhale_times else 0
    feats["avg_depth"] = np.mean(depths) if depths else 0
    feats["inhale_exhale_ratio"] = (feats["avg_inhale_time"] / feats["avg_exhale_time"]
                                    if feats["avg_exhale_time"] > 0 else 0)

    fft_vals = fft(filtered)
    freqs = fftfreq(len(filtered), 1/sr)
    pos = freqs > 0
    fft_mag = np.abs(fft_vals[pos])
    fft_freq = freqs[pos]

    if len(fft_mag) > 0:
        feats["dominant_frequency"] = fft_freq[np.argmax(fft_mag)]
        feats["spectral_energy"] = np.sum(fft_mag**2)
        feats["spectral_entropy"] = -np.sum(
            (fft_mag / np.sum(fft_mag)) * np.log2(fft_mag / np.sum(fft_mag) + 1e-10)
        )
        band = (fft_freq >= 0.1) & (fft_freq <= 0.35)
        feats["resp_band_power"] = np.sum(fft_mag[band]**2) if np.any(band) else 0
    else:
        feats["dominant_frequency"] = 0
        feats["spectral_energy"] = 0
        feats["spectral_entropy"] = 0
        feats["resp_band_power"] = 0

    return feats

LABEL_MAP = {1: "Baseline", 2: "Stress", 3: "Amusement"}

st.sidebar.header("Input")
mode = st.sidebar.radio("Choose input mode", ["Demo (synthetic)", "Upload .npy"])

signal = None

if mode == "Demo (synthetic)":
    emotion_choice = st.sidebar.selectbox("Simulate emotion", ["Baseline", "Stress", "Amusement"])
    params = {
        "Baseline": (21, 2.0, 0.05),
        "Stress": (14, 1.2, 0.08),
        "Amusement": (19, 1.8, 0.06)
    }
    rate, depth, noise = params[emotion_choice]
    t = np.linspace(0, 60, 700*60)
    signal = depth * np.sin(2 * np.pi * (rate/60) * t) + np.random.normal(0, noise, len(t))
    st.info(f"Simulated signal for: {emotion_choice} (rate={rate} bpm)")

else:
    uploaded = st.sidebar.file_uploader("Upload respiration signal (.npy, 700Hz)", type=["npy"])
    if uploaded is not None:
        signal = np.load(uploaded).flatten()
        st.success(f"Loaded signal: {signal.shape[0]} samples ({signal.shape[0]/700:.1f}s)")

if signal is not None:
    st.subheader("Signal preview")
    st.line_chart(signal[:min(10000, len(signal))])

    if st.button("Classify Emotion"):
        if len(signal) < 700 * 60:
            st.error(f"Need at least 60s of signal. Got {len(signal)/700:.1f}s")
        else:
            window = signal[:700*60].astype(float)
            feats = extract_features(window)
            feat_array = np.array([feats[name] for name in feature_names]).reshape(1, -1)
            feat_scaled = scaler.transform(feat_array)

            pred = int(model.predict(feat_scaled)[0])
            probs = model.predict_proba(feat_scaled)[0]

            st.subheader("Prediction")
            st.markdown(f"### Emotion: **{LABEL_MAP[pred]}**")

            prob_df = pd.DataFrame({
                "Emotion": [LABEL_MAP[i] for i in [1, 2, 3]],
                "Probability": [float(probs[i-1]) for i in [1, 2, 3]]
            })
            st.bar_chart(prob_df.set_index("Emotion"))

            with st.expander("Extracted features"):
                st.json({k: float(v) for k, v in feats.items()})
