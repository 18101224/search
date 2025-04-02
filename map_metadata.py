import json
import os
from pathlib import Path
from typing import Dict, Optional
import re

class MetadataMapper:
    def __init__(self, meta_dir: str):
        """Initialize the metadata mapper with a directory containing meta files.
        
        Args:
            meta_dir: Directory containing the meta files
        """
        self.meta_dir = Path(meta_dir)
        self.metadata_cache = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load all metadata files into memory."""
        for meta_file in self.meta_dir.glob("splice_*_raw.txt*"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            # Each line is a JSON object
                            data = json.loads(line.strip())
                            # Extract the base filename without extension
                            wav_filename = os.path.basename(data.get('wav_file', ''))
                            base_name = os.path.splitext(wav_filename)[0]
                            self.metadata_cache[base_name] = data
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error loading {meta_file}: {e}")
                continue
    
    def get_metadata(self, wav_file_path: str) -> Optional[Dict]:
        """Get metadata for a given WAV file path.
        
        Args:
            wav_file_path: Path to the WAV file
            
        Returns:
            Dictionary containing metadata or None if not found
        """
        # Extract the base filename without extension
        base_name = os.path.splitext(os.path.basename(wav_file_path))[0]
        return self.metadata_cache.get(base_name)
    
    def get_mp3_url(self, wav_file_path: str) -> Optional[str]:
        """Get the MP3 URL for a given WAV file path.
        
        Args:
            wav_file_path: Path to the WAV file
            
        Returns:
            MP3 URL or None if not found
        """
        metadata = self.get_metadata(wav_file_path)
        if metadata:
            return metadata.get('mp3_url')
        return None
    
    def download_mp3(self, wav_file_path: str, output_dir: str = "downloads") -> Optional[str]:
        """Download the MP3 file for a given WAV file path.
        
        Args:
            wav_file_path: Path to the WAV file
            output_dir: Directory to save the downloaded MP3 file
            
        Returns:
            Path to the downloaded MP3 file or None if download failed
        """
        mp3_url = self.get_mp3_url(wav_file_path)
        if not mp3_url:
            print(f"No MP3 URL found for {wav_file_path}")
            return None
            
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_file = output_dir / f"{os.path.splitext(os.path.basename(wav_file_path))[0]}.mp3"
        
        # Skip if file already exists
        if output_file.exists():
            print(f"File already exists: {output_file}")
            return str(output_file)
            
        try:
            import requests
            from tqdm import tqdm
            
            # Download the file with progress bar
            response = requests.get(mp3_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(output_file, 'wb') as f, tqdm(
                desc=f"Downloading {output_file.name}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
                    
            return str(output_file)
            
        except Exception as e:
            print(f"Error downloading MP3: {e}")
            return None

# Example usage:
if __name__ == "__main__":
    # Initialize the mapper with the meta directory
    mapper = MetadataMapper("search/meta")
    
    # Example WAV file path
    wav_file = "samples/55fed749ffb2ee2487a1d8d8b116fce88b5461b47e0dae1a3efeeea1f99a5ae9.wav"
    
    # Get metadata
    metadata = mapper.get_metadata(wav_file)
    if metadata:
        print("Metadata:", metadata)
        
        # Get MP3 URL
        mp3_url = mapper.get_mp3_url(wav_file)
        if mp3_url:
            print("MP3 URL:", mp3_url)
            
            # Download MP3
            output_file = mapper.download_mp3(wav_file)
            if output_file:
                print(f"Downloaded to: {output_file}")
    else:
        print("No metadata found for", wav_file) 