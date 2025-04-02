import os
import librosa
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import multiprocessing as mp
from functools import partial

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--samples_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker processes. Default: number of CPU cores')
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

def process_file(args):
    wav_file, output_dir, min_duration = args
    is_valid, reason = is_valid_audio(wav_file, min_duration=min_duration)
    
    if is_valid:
        # Copy valid file to output directory
        filename = os.path.basename(wav_file)
        new_path = os.path.join(output_dir, filename)
        os.system(f'cp "{wav_file}" "{new_path}"')
        return True, filename, reason
    else:
        return False, os.path.basename(wav_file), reason

def filter_samples(samples_dir, output_dir, min_duration=0.1, num_workers=None):
    # Get all wav files
    wav_files = []
    for root, dirs, files in os.walk(samples_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    process_args = [(wav_file, output_dir, min_duration) for wav_file in wav_files]
    
    # Set up multiprocessing pool
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    print(f"Processing audio files using {num_workers} workers...")
    
    # Process files in parallel
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_file, process_args),
            total=len(wav_files),
            desc="Processing files"
        ))
    
    # Count results
    valid_count = sum(1 for is_valid, _, _ in results if is_valid)
    invalid_count = sum(1 for is_valid, _, _ in results if not is_valid)
    
    # Print skipped files
    for is_valid, filename, reason in results:
        if not is_valid:
            print(f"\nSkipped {filename}. Reason: {reason}")
    
    print(f'\nProcessing complete:')
    print(f'Valid files: {valid_count}')
    print(f'Invalid files: {invalid_count}')

if __name__ == "__main__":
    args = get_args()
    filter_samples(args.samples_dir, args.output_dir, min_duration=0.5, num_workers=args.num_workers)
    
