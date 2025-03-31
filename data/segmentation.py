import os
import tarfile
import requests
from tqdm import tqdm
from pathlib import Path

def download_and_extract_dataset(url_list_path, output_dir, dataset_type='sa1b'):
    """Download and extract dataset from list of CDN links"""
    # Read URLs from file
    with open(url_list_path) as f:
        lines = [line.strip().split('\t') for line in f.readlines()[1:]]  # Skip header
        
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download and process each file
    for filename, url in tqdm(lines, desc=f'Processing {dataset_type}'):
        # Download file
        dest_path = Path(output_dir) / filename
        if not dest_path.exists():
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=filename
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))
        
        # Extract tar file
        print(f"Extracting {filename}...")
        with tarfile.open(dest_path) as tar:
            tar.extractall(path=output_dir)
        
        # Cleanup (optional)
        # dest_path.unlink()

# Example usage:
# download_and_extract_dataset('SA-1B_dataset_copy.txt', 'sa1b_data')
# download_and_extract_dataset('SA-V_dataset_copy.txt', 'sav_data')