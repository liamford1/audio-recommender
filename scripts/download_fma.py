import os
import requests
import zipfile
from pathlib import Path

def download_fma_small():
    """Download and extract FMA small dataset"""
    
    # Create directories
    data_dir = Path("data/raw/fma_small")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # FMA small dataset URL
    url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    zip_path = "data/raw/fma_small.zip"
    
    print("📥 Downloading FMA small dataset (~7.2GB)...")
    print("This will take several minutes...")
    
    # Download with progress
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    print(f"\n✅ Download complete: {zip_path}")
    
    # Extract
    print("📂 Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/raw/")
    
    print("✅ FMA dataset ready!")
    print(f"Files extracted to: {data_dir}")
    
    # Clean up zip file
    os.remove(zip_path)
    print("🗑️  Cleaned up zip file")

if __name__ == "__main__":
    download_fma_small()