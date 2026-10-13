# Breathing Pattern Based Emotion Classification

Classifies emotions (Baseline, Stress, Amusement) from respiration signals using Machine Learning.

## Overview

Mental health monitoring is often reactive rather than proactive. This project explores whether breathing patterns can be used to detect emotional states passively and in real time.

Using the WESAD dataset, respiration signals from a chest-worn RespiBAN sensor (700 Hz) are processed into 17 time-domain and frequency-domain features. Four ML models are trained and compared, with the best one deployed as an interactive web application.

## Results

| Model | Test Accuracy | F1-Score | CV Accuracy |
|-------|---------------|----------|-------------|
| Random Forest | 89.74% | 0.9023 | 88.12% |
| Gradient Boosting | 84.62% | 0.8601 | 90.20% |
| SVM | 82.05% | 0.8240 | 84.53% |
| KNN | 82.05% | 0.8255 | 86.09% |

Best model: Random Forest - 89.74% test accuracy, 0.90 F1-score

## Approach

- Data: WESAD (Wearable Stress and Affect Detection)
- Signal: Chest respiration (RespiBAN, 700 Hz)
- Subject: S2 (109 Baseline, 55 Stress, 30 Amusement windows)
- Window: 60 seconds with 10-second step

### Features (17)

- Statistical: mean, std, max, min, range, rms, slope
- Breathing: breathing_rate, breathing_rate_std
- Cycle: avg_inhale_time, avg_exhale_time, avg_depth, inhale_exhale_ratio
- Spectral: dominant_frequency, spectral_energy, spectral_entropy, resp_band_power

### Signal Processing

- Bandpass filter 0.1-0.35 Hz using sosfiltfilt
- Peak detection for breathing cycle analysis
- FFT for spectral features

### Models

- Random Forest
- Gradient Boosting
- SVM
- KNN

## Tech Stack

Python, Scikit-Learn, SciPy, Streamlit, Pandas, NumPy, Joblib

## Local Setup

git clone https://github.com/ijugunu/breathing-pattern-emotion-classification.git
cd breathing-pattern-emotion-classification
pip install -r requirements.txt
streamlit run app.py

## Project Structure

app.py
best_model.pkl
scaler.pkl
feature_names.txt
requirements.txt
README.md

## Reference

Schmidt et al. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. ICMI 2018.

## License

MIT License

