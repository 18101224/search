import librosa, os, torch
import numpy as np

def compute_fft_descriptors_internal(y, sr):
    """기존 FFT 특성 추출 함수 (내부용)"""
    # 기존 FFT 처리 코드
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1 / sr)

    # Compute amplitudes
    amplitudes = np.abs(fft)
    sum_amplitudes = np.sum(amplitudes)

    # 진폭이 0인 경우 처리
    if sum_amplitudes == 0:
        return np.zeros(7)  # 모든 특성을 0으로 반환

    # Fundamental frequency (f0) estimation
    f0_idx = np.argmax(amplitudes)
    f0 = max(freqs[f0_idx], 1e-6)  # 0으로 나누기 방지

    # Affinity (A)
    A = np.sum(amplitudes * freqs) / (f0 * sum_amplitudes + 1e-6)

    # Sharpness (S)
    S = amplitudes[f0_idx] / (sum_amplitudes + 1e-6)

    # Harmonicity (H)
    H = np.sum((freqs / f0 - np.round(freqs / f0)) * amplitudes) / (sum_amplitudes + 1e-6)

    # Monotony (M)
    freq_diff = np.diff(freqs)
    # 0으로 나누기 방지
    freq_diff = np.where(freq_diff == 0, 1e-6, freq_diff)
    M = f0 / len(amplitudes) * np.sum(np.diff(amplitudes) / freq_diff)

    # Mean Affinity (MA)
    MA = np.sum(np.abs(freqs - np.mean(freqs))) / (len(freqs) * f0 + 1e-6)

    # Mean Contrast (MC)
    MC = np.sum(np.abs(amplitudes[0] - amplitudes)) / (len(amplitudes) + 1e-6)

    # NaN 값 체크 및 대체
    features = np.array([f0, A, S, H, M, MA, MC])
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    return features

def compute_enhanced_descriptors(path):
    """향상된 오디오 특성 추출 함수"""
    is_mp3 = path.lower().endswith('.mp3')
    temp_wav_path = None

    try:
        audio_path = path
        # 오디오 로드
        y, sr = librosa.load(audio_path, sr=None)

        # 임시 파일 정리
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

        # 기존 FFT 특성
        fft_features = compute_fft_descriptors_internal(y, sr)

        # 1. MFCC (Mel-Frequency Cepstral Coefficients) - 음색 표현에 효과적
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)

        # 2. 스펙트럼 대비(Spectral Contrast) - 음색의 피크와 밸리 관계 포착
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        # 3. 스펙트럼 중심(Spectral Centroid) - 스펙트럼의 "무게 중심"
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroid_mean = np.mean(centroid)

        # 4. 스펙트럼 대역폭(Spectral Bandwidth) - 주파수 분포 폭
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        bandwidth_mean = np.mean(bandwidth)

        # 5. 퍼커시브 특성(Percussive Features) - 드럼 FX에 특화
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        # 타악기 성분의 에너지
        percussive_energy = np.sum(y_percussive ** 2)
        # 타악기/하모닉 비율
        perc_harm_ratio = np.sum(y_percussive ** 2) / np.sum(y_harmonic ** 2) if np.sum(y_harmonic ** 2) > 0 else 1.0

        # 6. 시간적 특성(Temporal Features)
        # 소리의 어택(Attack) 특성
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_mean = np.mean(onset_env)
        onset_max = np.max(onset_env)

        # 7. 변화율(Flux) - 인접 프레임간 스펙트럼 변화
        spec = np.abs(librosa.stft(y))
        flux = np.sum(np.diff(spec, axis=1) ** 2, axis=0)
        flux_mean = np.mean(flux)

        # 모든 특성 결합 및 정규화
        features = np.concatenate([
            fft_features,  # 기존 7개 특성
            mfcc_mean, mfcc_var,  # MFCC 평균 및 분산 (26개)
            contrast_mean,  # 스펙트럼 대비 (7개)
            [centroid_mean, bandwidth_mean],  # 중심 및 대역폭 (2개)
            [percussive_energy, perc_harm_ratio],  # 퍼커시브 특성 (2개)
            [onset_mean, onset_max, flux_mean]  # 시간적 특성 (3개)
        ])

        return torch.tensor(features)

    except Exception as e:
        return False
