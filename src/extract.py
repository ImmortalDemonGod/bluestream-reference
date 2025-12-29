"""
extract.py - Download data from EPA Water Quality Portal

Your task: Implement the download function
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
    
    EPA API Documentation: https://www.waterqualitydata.us/webservices_documentation/
    
    TODO: Implement this function
    
    Hints:
    - Base URL: "https://www.waterqualitydata.us/data/Result/search"
    - Query parameters needed:
      * statecode: From config
      * characteristicName: From config
      * startDateLo, startDateHi: From config
      * mimeType: 'csv'
      * zip: 'yes'
    - Use requests.get() with stream=True
    - EPA returns a ZIP file containing CSV
    - Extract the CSV that has 'result' in the filename
    - Rename to 'oklahoma_chloride.csv'
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to downloaded CSV file
        
    Expected output:
        - File: data/raw/oklahoma_chloride.csv
        - Size: ~75 MB
        - Records: ~155,000
    """
    
    # TODO: Build API URL and parameters
    base_url = "https://www.waterqualitydata.us/data/Result/search"
    params = {
        # TODO: Fill in query parameters from config
    }
    
    # TODO: Create output directory
    output_dir = Path(config['output_paths']['raw_data'])
    # HINT: Use output_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Download the zip file
    print(f"Downloading {config['data_sources']['characteristic']} data from EPA...")
    
    # HINT: response = requests.get(base_url, params=params, stream=True)
    # HINT: Check response.raise_for_status() to catch errors
    
    # TODO: Save zip file temporarily
    zip_path = output_dir / "oklahoma_data.zip"
    # HINT: Iterate through response.iter_content(chunk_size=8192) and write chunks
    
    # TODO: Extract CSV from zip
    # HINT: Use zipfile.ZipFile to open the zip
    # HINT: Look for files with 'result' in name and ending with '.csv'
    # HINT: Extract and rename to 'oklahoma_chloride.csv'
    
    # TODO: Clean up zip file
    # HINT: zip_path.unlink()
    
    # TODO: Verify the downloaded file
    final_path = output_dir / "oklahoma_chloride.csv"
    # HINT: Load with pd.read_csv(final_path, low_memory=False)
    # HINT: Print record count and file size
    
    return final_path

def main():
    """Main execution"""
    config = load_config()
    filepath = download_oklahoma_chloride(config)
    print("\n✅ Data extraction complete")

if __name__ == "__main__":
    main()
