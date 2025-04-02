import numpy as np
import librosa
import torch
import torchaudio
from .vanilla import compute_fft_descriptors


def compute_mfcc_cpu(path, n_mfcc=13):
    """CPU implementation of MFCC computation with FFT concatenation"""
    try:
        # Get the 7D FFT vector
        fft_vector = compute_fft_descriptors_cpu(path)
        if fft_vector is False:
            return False
            
        y, sr = librosa.load(path, sr=None)
    except:
        return False
    
    # Compute MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Compute delta and delta-delta
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Concatenate features
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    
    # Compute statistics
    mean = np.mean(features, axis=1)
    std = np.std(features, axis=1)
    max_val = np.max(features, axis=1)
    min_val = np.min(features, axis=1)
    
    # Combine all statistics
    mfcc_stats = np.concatenate([mean, std, max_val, min_val])
    
    # Concatenate with FFT vector
    combined_features = np.concatenate([fft_vector.numpy(), mfcc_stats])
    
    return torch.tensor(combined_features)

def compute_mfcc_cuda(path, n_mfcc=13):
    """CUDA implementation of MFCC computation with FFT concatenation"""
    try:
        # Get the 7D FFT vector
        fft_vector = compute_fft_descriptors_cuda(path)
        if fft_vector is False:
            return False
            
        # Load audio using torchaudio
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
    except:
        return False
    
    # Move to GPU
    waveform = waveform.cuda()
    
    # Compute MFCC
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=n_mfcc,
        melkwargs={'n_fft': 2048, 'hop_length': 512, 'mel_channels': 128}
    ).cuda()
    
    mfcc = mfcc_transform(waveform)
    
    # Compute delta and delta-delta
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    
    # Concatenate features
    features = torch.cat([mfcc, delta, delta2], dim=1)
    
    # Compute statistics
    mean = torch.mean(features, dim=1)
    std = torch.std(features, dim=1)
    max_val = torch.max(features, dim=1)[0]
    min_val = torch.min(features, dim=1)[0]
    
    # Combine all statistics
    mfcc_stats = torch.cat([mean, std, max_val, min_val])
    
    # Concatenate with FFT vector
    combined_features = torch.cat([fft_vector, mfcc_stats])
    
    return combined_features

def compute_mfcc(path, n_mfcc=13, use_cuda=True):
    """
    Compute combined features (FFT + MFCC) using either CPU or CUDA
    
    Args:
        path (str): Path to audio file
        n_mfcc (int): Number of MFCC coefficients
        use_cuda (bool): Whether to use CUDA if available
    
    Returns:
        torch.Tensor: Combined feature vector (7D FFT + MFCC statistics)
    """
    
    if use_cuda and torch.cuda.is_available():
        return compute_mfcc_cuda(path, n_mfcc)
    return compute_mfcc_cpu(path, n_mfcc) 