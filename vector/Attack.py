import numpy as np
import librosa
import scipy.signal
import os


def extract_attack_features(wav_path, sr=22050):
    # Load audio
    y, sr = librosa.load(wav_path, sr=sr)

    # Step 1: Full-wave rectification (절댓값)
    y_rectified = np.abs(y)

    # Step 2: Low-pass filter to extract envelope
    b, a = scipy.signal.butter(2, 5 / (sr / 2), btype='low')  # 5 Hz cutoff
    envelope = scipy.signal.filtfilt(b, a, y_rectified)

    # Step 3: Normalize envelope
    envelope = envelope / np.max(envelope + 1e-7)

    # Step 4: Derivative (slope)
    derivative = np.diff(envelope)

    # Step 5: Find attack region (where derivative is high)
    peak_index = np.argmax(envelope)
    attack_start_idx = np.argmax(derivative[:peak_index])
    attack_end_idx = peak_index

    # Step 6: Feature extraction
    attack_time = (attack_end_idx - attack_start_idx) / sr  # seconds
    attack_slope = (envelope[attack_end_idx] - envelope[attack_start_idx]) / (attack_end_idx - attack_start_idx + 1e-7)

    return {
        "attack_time": attack_time,
        "attack_slope": attack_slope
    }