import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from scipy.signal import find_peaks, butter, sosfiltfilt, resample
from scipy.fft import fft, fftfreq

st.set_page_config(page_title="Breathing Emotion Classifier", layout="wide")

st.title("Breathing Pattern Emotion Classification")
st.markdown("Classifies emotions (Baseline / Stress / Amusement) from respiration signals.")


@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_names.txt") as f:
        feature_names = [line.strip() for line in f if line.strip()]
    return model, scaler, feature_names


model, scaler, feature_names = load_artifacts()

SOS = butter(4, [0.1 / (700 / 2), 0.35 / (700 / 2)], btype="band", output="sos")


def extract_features(signal, sr=700):
    signal = signal - np.mean(signal)
    filtered = sosfiltfilt(SOS, signal)

    feats = {
        "mean": float(np.mean(filtered)),
        "std": float(np.std(filtered)),
        "max": float(np.max(filtered)),
        "min": float(np.min(filtered)),
        "range": float(np.max(filtered) - np.min(filtered)),
        "rms": float(np.sqrt(np.mean(filtered ** 2))),
        "slope": float(np.polyfit(range(len(filtered)), filtered, 1)[0]),
    }

    peaks, _ = find_peaks(filtered, distance=sr * 0.3)
    troughs, _ = find_peaks(-filtered, distance=sr * 0.3)

    if len(peaks) > 1:
        intervals = np.diff(peaks) / sr
        feats["breathing_rate"] = float(60 / np.mean(intervals))
        feats["breathing_rate_std"] = float(60 * np.std(intervals))
    else:
        feats["breathing_rate"] = 0.0
        feats["breathing_rate_std"] = 0.0

    inhale_times, exhale_times, depths = [], [], []
    for i in range(min(len(peaks) - 1, len(troughs))):
        if peaks[i] < troughs[i]:
            inhale_times.append((troughs[i] - peaks[i]) / sr)
            exhale_times.append((peaks[i + 1] - troughs[i]) / sr)
            depths.append(filtered[peaks[i]] - filtered[troughs[i]])

    feats["avg_inhale_time"] = float(np.mean(inhale_times)) if inhale_times else 0.0
    feats["avg_exhale_time"] = float(np.mean(exhale_times)) if exhale_times else 0.0
    feats["avg_depth"] = float(np.mean(depths)) if depths else 0.0
    feats["inhale_exhale_ratio"] = (
        float(feats["avg_inhale_time"] / feats["avg_exhale_time"])
        if feats["avg_exhale_time"] > 0
        else 0.0
    )

    fft_vals = fft(filtered)
    freqs = fftfreq(len(filtered), 1 / sr)
    pos = freqs > 0
    fft_mag = np.abs(fft_vals[pos])
    fft_freq = freqs[pos]

    if len(fft_mag) > 0:
        feats["dominant_frequency"] = float(fft_freq[np.argmax(fft_mag)])
        feats["spectral_energy"] = float(np.sum(fft_mag ** 2))
        feats["spectral_entropy"] = float(
            -np.sum(
                (fft_mag / np.sum(fft_mag))
                * np.log2(fft_mag / np.sum(fft_mag) + 1e-10)
            )
        )
        band = (fft_freq >= 0.1) & (fft_freq <= 0.35)
        feats["resp_band_power"] = float(np.sum(fft_mag[band] ** 2)) if np.any(band) else 0.0
    else:
        feats["dominant_frequency"] = 0.0
        feats["spectral_energy"] = 0.0
        feats["spectral_entropy"] = 0.0
        feats["resp_band_power"] = 0.0

    return feats


LABEL_MAP = {1: "Baseline", 2: "Stress", 3: "Amusement"}

st.sidebar.header("Input")
mode = st.sidebar.radio(
    "Choose input mode",
    ["Demo (real WESAD sample)", "Upload CSV (respiration signal)", "Upload .npy"],
)

signal = None
detected_sr_global = 700

if mode == "Demo (real WESAD sample)":
    demo_class = st.sidebar.selectbox(
        "Pick a real WESAD window", ["Baseline", "Stress", "Amusement"]
    )
    file_map = {
        "Baseline": "wesad_sample_1.npy",
        "Stress": "wesad_sample_2.npy",
        "Amusement": "wesad_sample_3.npy",
    }
    fname = file_map[demo_class]
    if os.path.exists(fname):
        signal = np.load(fname).flatten()
        st.info(f"Loaded real WESAD window: {demo_class}")
    else:
        st.error(f"Missing file: {fname}")

elif mode == "Upload CSV (respiration signal)":
    uploaded = st.sidebar.file_uploader("Upload CSV (time, resp)", type=["csv"])

    with st.sidebar.expander("CSV format guide"):
        st.markdown("Your CSV should have 2 columns: time (seconds) and resp (respiration amplitude). Minimum 60 seconds required.")

    if uploaded is not None:
        try:
            user_df = pd.read_csv(uploaded)
            st.write("Preview:", user_df.head())

            col_names = [c.lower() for c in user_df.columns]
            if "time" in col_names and "resp" in col_names:
                time_col = user_df.columns[col_names.index("time")]
                resp_col = user_df.columns[col_names.index("resp")]
            else:
                time_col = user_df.columns[0]
                resp_col = user_df.columns[1]

            time_vals = user_df[time_col].values
            resp_vals = user_df[resp_col].values.astype(float)

            dt = np.median(np.diff(time_vals))
            detected_sr = int(round(1.0 / dt))

            st.success(
                f"Detected sampling rate: {detected_sr} Hz | "
                f"Duration: {time_vals[-1]:.1f}s | Samples: {len(resp_vals)}"
            )

            signal = resp_vals
            detected_sr_global = detected_sr
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

else:
    uploaded = st.sidebar.file_uploader("Upload .npy (700Hz respiration)", type=["npy"])
    if uploaded is not None:
        signal = np.load(uploaded).flatten()
        st.success(f"Loaded: {len(signal)} samples")


if signal is not None:
    st.subheader("Signal preview")
    st.line_chart(signal[: min(10000, len(signal))])

    if st.button("Classify Emotion"):
        sr_to_use = 700
        if mode == "Upload CSV (respiration signal)":
            sr_to_use = detected_sr_global

        min_samples = sr_to_use * 60

        if len(signal) < min_samples:
            st.error(
                f"Need at least 60s of signal at {sr_to_use} Hz. "
                f"Got {len(signal)} samples ({len(signal) / sr_to_use:.1f}s)."
            )
        else:
            window = signal[:min_samples].astype(float)

            if sr_to_use != 700:
                target_len = int(700 * 60)
                window = resample(window, target_len)

            feats = extract_features(window, sr=700)
            feat_array = np.array([feats[name] for name in feature_names]).reshape(1, -1)
            feat_scaled = scaler.transform(feat_array)

            pred = int(model.predict(feat_scaled)[0])
            probs = model.predict_proba(feat_scaled)[0]

            st.subheader("Prediction")
            st.markdown(f"### Emotion: **{LABEL_MAP[pred]}**")

            prob_df = pd.DataFrame(
                {
                    "Emotion": [LABEL_MAP[i] for i in [1, 2, 3]],
                    "Probability": [float(probs[i - 1]) for i in [1, 2, 3]],
                }
            )
            st.bar_chart(prob_df.set_index("Emotion"))

            with st.expander("Extracted features"):
                st.json(feats)
