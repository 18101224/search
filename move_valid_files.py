import os
import shutil
import librosa
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--source_dir', type=str)
    parser.add_argument('--dest_dir', type=str)
    parser.add_argument('--min_duration', type=float, default=0.5)
    parser.add_argument('--min_rms', type=float, default=0.05)
    args = parser.parse_args()
    return args

def is_valid_audio(file_path, min_duration=0.1, min_rms=0.01):
    """
    Check if an audio file is valid:
    - Not too short (default: > 0.1 seconds)
    - Has actual sound (default: RMS > 0.01)
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path)
        
        # Check duration
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < min_duration:
            return False, "Too short"
        
        # Check if there's actual sound
        rms = librosa.feature.rms(y=y)[0]
        if np.max(rms) < min_rms:
            return False, "No sound"
            
        return True, "Valid"
    except Exception as e:
        return False, f"Error: {str(e)}"

def move_valid_files(source_dir, dest_dir, min_duration=0.1, min_rms=0.01):
    """
    Move valid audio files from source directory to destination directory
    
    Args:
        source_dir (str): Directory containing audio files
        dest_dir (str): Directory to move valid files to
        min_duration (float): Minimum duration in seconds
        min_rms (float): Minimum RMS energy threshold
    """
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all wav files
    wav_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    print(f"Found {len(wav_files)} WAV files")
    
    # Process each file
    valid_count = 0
    invalid_count = 0
    
    for wav_file in tqdm(wav_files, desc="Processing files"):
        is_valid, reason = is_valid_audio(wav_file, min_duration, min_rms)
        
        if is_valid:
            # Move valid file to destination
            filename = os.path.basename(wav_file)
            dest_path = os.path.join(dest_dir, filename)
            
            # Handle duplicate filenames
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(dest_dir, f"{base}_{counter}{ext}")
                    counter += 1
            
            shutil.move(wav_file, dest_path)
            valid_count += 1
        else:
            invalid_count += 1
            print(f"\nInvalid file: {wav_file} - {reason}")
    
    print(f"\nProcessing complete:")
    print(f"Valid files moved: {valid_count}")
    print(f"Invalid files: {invalid_count}")

if __name__ == "__main__":
    args = get_args()
    source_dir = args.source_dir
    dest_dir = args.dest_dir
    
    # You can adjust these thresholds
    min_duration = 0.1  # Minimum duration in seconds
    min_rms = 0.01     # Minimum RMS energy threshold
    
    move_valid_files(source_dir, dest_dir, min_duration, min_rms) 