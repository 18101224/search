import numpy as np
import librosa
import torch

def compute_fft_descriptors_cpu(path):
    """CPU implementation of FFT descriptors"""
    try:
        y, sr = librosa.load(path, sr=None)
    except:
        return False
    
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

    return torch.tensor([f0, A, S, H, M, MA, MC])

def compute_fft_descriptors_cuda(path):
    """CUDA implementation of FFT descriptors"""
    try:
        y, sr = librosa.load(path, sr=None)
    except:
        return False
    
    # Move data to GPU
    y = torch.from_numpy(y).cuda() 
    
    # Compute FFT
    fft = torch.fft.rfft(y)
    freqs = torch.from_numpy(np.fft.rfftfreq(len(y), 1/sr)).cuda()

    # Compute amplitudes
    amplitudes = torch.abs(fft)

    # Fundamental frequency (f0) estimation
    f0 = freqs[torch.argmax(amplitudes)]

    # Affinity (A)
    A = torch.sum(amplitudes * freqs) / (f0 * torch.sum(amplitudes))

    # Sharpness (S)
    S = amplitudes[torch.argmax(amplitudes)] / torch.sum(amplitudes)

    # Harmonicity (H)
    H = torch.sum((freqs / f0 - torch.round(freqs / f0)) * amplitudes) / torch.sum(amplitudes)

    # Monotony (M)
    M = f0 / len(amplitudes) * torch.sum(torch.diff(amplitudes) / torch.diff(freqs))

    # Mean Affinity (MA)
    MA = torch.sum(torch.abs(freqs - torch.mean(freqs))) / (len(freqs) * f0)

    # Mean Contrast (MC)
    MC = torch.sum(torch.abs(amplitudes[0] - amplitudes)) / len(amplitudes)

    return torch.tensor([f0, A, S, H, M, MA, MC])

def compute_fft_descriptors(path, use_cuda=True):
    """
    Compute FFT descriptors using either CPU or CUDA
    
    Args:
        path (str): Path to audio file
        use_cuda (bool): Whether to use CUDA if available
    
    Returns:
        torch.Tensor: 7-dimensional feature vector
    """
    if use_cuda and torch.cuda.is_available():
        return compute_fft_descriptors_cuda(path)
    return compute_fft_descriptors_cpu(path)

