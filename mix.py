import librosa
import sounddevice as sd  # or use simpleaudio
import soundfile as sf

def mix_samples(drum_file, guitar_file, output_file="mixed.wav", sr=44100, drum_level=1.0, guitar_level=1.0):
    """
    Mixes two audio samples (drum and guitar) and saves the result to a WAV file.

    Args:
        drum_file (str): Path to the drum sample WAV file.
        guitar_file (str): Path to the guitar sample WAV file.
        output_file (str, optional): Path to save the mixed WAV file. Defaults to "mixed.wav".
        sr (int, optional): Sample rate for resampling. Defaults to 44100.
        drum_level (float, optional): Volume level for the drum sample (0.0 to 1.0). Defaults to 1.0.
        guitar_level (float, optional): Volume level for the guitar sample (0.0 to 1.0). Defaults to 1.0.
    """
        # Load audio files using librosa
    drum, sr_drum = librosa.load(drum_file, sr=sr)
    guitar, sr_guitar = librosa.load(guitar_file, sr=sr)

    # Resample if necessary (ensure both samples have the same sample rate)
    if sr_drum != sr:
        drum = librosa.resample(drum, orig_sr=sr_drum, target_sr=sr)
    if sr_guitar != sr:
        guitar = librosa.resample(guitar, orig_sr=sr_guitar, target_sr=sr)

    length = min(len(drum), len(guitar))
    drum = drum[:length]
    guitar = guitar[:length]
    # Mix the samples
    mixed = drum_level * drum + guitar_level * guitar

    # Normalize the mixed audio to prevent clipping
    mixed /= np.max(np.abs(mixed))

    # Save the mixed audio to a WAV file
    sf.write(output_file,mixed,sr)
    print(f"Successfully mixed audio and saved to {output_file}")

import numpy as np
import librosa.display
import matplotlib.pyplot as plt

def plot_audio(samples, sample_rate, title):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y=samples, sr=sample_rate)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

# Example Usage
drum_file = "lp/mix_drum.wav"  # Replace with your drum sample file
guitar_file = "lp/mix_guitar.wav" # Replace with your guitar sample file
output_file = "mixed_audio.wav"
sample_rate = 44100

# Call the function to mix the audio
mix_samples(drum_file, guitar_file, output_file, sample_rate, drum_level = 0.8, guitar_level = 0.6)
# Load the mixed file and plot the waveform
mixed, sr = librosa.load(output_file, sr=sample_rate)

plot_audio(mixed, sr, "Waveform of mixed samples")

#Playing of audio samples

def play_audio(samples, sample_rate):
    """Plays the given audio samples."""
    try:
        sd.play(samples, sample_rate)
        sd.wait()  # Wait until the sound finishes playing
    except Exception as e:
        print(f"Error playing audio: {e}")

play_audio(mixed, sample_rate)
