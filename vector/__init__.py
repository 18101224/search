import numpy as np
import librosa, torch

def norm(vec):
    vec2 = vec*vec
    norm = np.sqrt(sum(vec2))
    return torch.tensor(vec/norm)


def compute_fft_descriptors(path):
    y, sr = librosa.load(path,sr=None)
    # Compute FFT
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1/sr)


    # Compute amplitudes
    amplitudes = np.abs(fft)

    # Fundamental frequency (f0) estimation
    f0 = freqs[np.argmax(amplitudes)]

    # Affinity (A)
    A = np.sum(amplitudes * freqs) / (f0 * np.sum(amplitudes))

    # Sharpness (S)
    S = amplitudes[np.argmax(amplitudes)] / np.sum(amplitudes)

    # Harmonicity (H)
    H = np.sum((freqs / f0 - np.round(freqs / f0)) * amplitudes) / np.sum(amplitudes)

    # Monotony (M)
    M = f0 / len(amplitudes) * np.sum(np.diff(amplitudes) / np.diff(freqs))

    # Mean Affinity (MA)
    MA = np.sum(np.abs(freqs - np.mean(freqs))) / (len(freqs) * f0)

    # Mean Contrast (MC)
    MC = np.sum(np.abs(amplitudes[0] - amplitudes)) / len(amplitudes)

    return norm(np.array([f0, A, S, H, M, MA, MC]))

