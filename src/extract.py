"""
extract.py - Download data from EPA Water Quality Portal

Expected runtime: 5-10 minutes
Expected output: data/raw/oklahoma_chloride.csv (~155,000 records, ~75 MB)
"""

import requests
import zipfile
import pandas as pd
from pathlib import Path
import yaml

def load_config():
    """Load configuration from YAML file"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def download_oklahoma_chloride(config):
    """
    Download Oklahoma chloride data from EPA Water Quality Portal
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to downloaded CSV file
    """
    
    # EPA WQP API endpoint
    base_url = "https://www.waterqualitydata.us/data/Result/search"
    
    # Query parameters
    params = {
        'statecode': config['data_sources']['state_code'],
        'characteristicName': config['data_sources']['characteristic'],
        'startDateLo': config['data_sources']['date_range']['start'],
        'startDateHi': config['data_sources']['date_range']['end'],
        'mimeType': 'csv',
        'zip': 'yes',
        'dataProfile': 'resultPhysChem'
    }
    
    # Create output directory
    output_dir = Path(config['output_paths']['raw_data'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {config['data_sources']['characteristic']} data from EPA...")
    print(f"Date range: {params['startDateLo']} to {params['startDateHi']}")
    
    # Download data
    response = requests.get(base_url, params=params, stream=True)
    response.raise_for_status()
    
    # Save zip file
    zip_path = output_dir / "oklahoma_data.zip"
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Download complete: {zip_path}")
    
    # Extract CSV from zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Find the result CSV file
        csv_files = [f for f in zip_ref.namelist() if 'result' in f.lower() and f.endswith('.csv')]
        if not csv_files:
            raise ValueError("No result CSV found in downloaded zip")
        
        # Extract to raw data directory
        csv_filename = csv_files[0]
        zip_ref.extract(csv_filename, output_dir)
        
        # Rename to standard name
        extracted_path = output_dir / csv_filename
        final_path = output_dir / "oklahoma_chloride.csv"
        extracted_path.rename(final_path)
    
    # Clean up zip file
    zip_path.unlink()
    
    # Verify file
    df = pd.read_csv(final_path, low_memory=False)
    print(f"\nDownload successful!")
    print(f"  Records: {len(df):,}")
    print(f"  File: {final_path}")
    print(f"  Size: {final_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return final_path

def main():
    """Main execution"""
    config = load_config()
    filepath = download_oklahoma_chloride(config)
    print("\n✅ Data extraction complete")

if __name__ == "__main__":
    main()
