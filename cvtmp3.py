from glob import glob 
import soundfile as sf
import librosa, os 
from tqdm import tqdm 
from argparse import ArgumentParser
import concurrent.futures
from pathlib import Path
def convert_mp3_to_wav(mp3_path, wav_dir):
    """Convert a single MP3 file to WAV format"""
    try:
        name = Path(mp3_path).stem
        y, sr = librosa.load(mp3_path, sr=None)
        sf.write(f'{wav_dir}/{name}.wav', y, sr)
        os.remove(mp3_path)
        return True, mp3_path
    except Exception as e:
        return False, mp3_path

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--mp3_dir', type=str)
    parser.add_argument('--wav_dir', type=str)
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to use')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    mp3_dir = args.mp3_dir
    wav_dir = args.wav_dir
    num_threads = args.num_threads
    
    # Create output directory if it doesn't exist
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    
    # Get all MP3 files
    mp3_files = glob(f'{mp3_dir}/*.mp3')
    total_files = len(mp3_files)
    print(f"Found {total_files} MP3 files")


    # Initialize counters
    success_count = 0
    fail_count = 0
    
    # Process files using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(convert_mp3_to_wav, mp3_path, wav_dir): mp3_path 
            for mp3_path in mp3_files
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=total_files, desc="Converting files") as pbar:
            for future in concurrent.futures.as_completed(future_to_file):
                success, file_path = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"\nFailed to convert: {file_path}")
                pbar.update(1)
    
    print(f"\nConversion complete:")
    print(f"Total files: {total_files}")
    print(f"Successful conversions: {success_count}")
    print(f"Failed conversions: {fail_count}")