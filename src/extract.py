"""
extract.py - Download data from EPA Water Quality Portal and OCC ArcGIS API

Expected runtime: 5-10 minutes
Expected output:
  data/raw/oklahoma_chloride.csv (~50,000 records, ~32 MB)
  data/raw/arcgis_volunteer_chloride.csv (Blue Thumb volunteer data from ArcGIS, 2015+)
"""

import hashlib
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
        'siteType': config['data_sources']['site_type'],
        'sampleMedia': config['data_sources']['sample_media'],
        'providers': config['data_sources'].get('providers'),
        'startDateLo': config['data_sources']['date_range']['start'],
        'startDateHi': config['data_sources']['date_range']['end'],
        'mimeType': 'csv',
        'zip': 'yes',
        'dataProfile': 'resultPhysChem'
    }
    
    # Create output directory
    output_dir = Path(config['output_paths']['raw_data'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading data from EPA...")
    print(f"Characteristics: {params['characteristicName']}")
    print(f"Site Type: {params['siteType']}")
    print(f"Sample Media: {params['sampleMedia']}")
    print(f"Providers: {params['providers']}")
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

ARCGIS_BASE_URL = (
    "https://services5.arcgis.com/L6JGkSUcgPo1zSDi/arcgis/rest/services/"
    "bluethumb_oct2020_view/FeatureServer/0/query"
)
ARCGIS_FIELDS = [
    "WBIDName", "lat", "lon", "day",
    "chloridetest1", "chloridetest3", "chloridetest5",
    "Chloride_Low1_Final", "QAQC_Complete",
]
ARCGIS_PAGE_SIZE = 1000


def download_arcgis_volunteer_data(config):
    """
    Download Blue Thumb volunteer chloride data from OCC's public ArcGIS REST API.

    Covers 2015+ data only. Paginates through all records (max 1000 per request).
    Saves raw response to data/raw/arcgis_volunteer_chloride.csv.

    Note: The static R-Shiny CSV (data/Requested_Blue Thumb Chemical Data.csv)
    contains pre-2015 data not available via ArcGIS. This fetch supplements but
    does not fully replace the static CSV.

    Returns:
        Path to saved CSV, or None if fetch fails
    """
    output_dir = Path(config['output_paths']['raw_data'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "arcgis_volunteer_chloride.csv"

    print(f"\nDownloading Blue Thumb volunteer data from ArcGIS...")
    print(f"  Endpoint: {ARCGIS_BASE_URL}")
    print(f"  Fields: {', '.join(ARCGIS_FIELDS)}")

    all_records = []
    offset = 0

    while True:
        params = {
            "where": "Chloride_Low1_Final IS NOT NULL",
            "outFields": ",".join(ARCGIS_FIELDS),
            "f": "json",
            "orderByFields": "day ASC",
            "resultOffset": offset,
            "resultRecordCount": ARCGIS_PAGE_SIZE,
        }

        try:
            response = requests.get(ARCGIS_BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"  WARNING: ArcGIS fetch failed at offset {offset}: {e}")
            print(f"  The pipeline will use the static Blue Thumb CSV instead.")
            return None

        if "error" in data:
            print(f"  WARNING: ArcGIS returned error: {data['error']}")
            return None

        features = data.get("features", [])
        if not features:
            break

        for f in features:
            attrs = f.get("attributes", {})
            if attrs:
                all_records.append(attrs)

        print(f"  Fetched {len(all_records):,} records (page {offset // ARCGIS_PAGE_SIZE + 1})...")

        if len(features) < ARCGIS_PAGE_SIZE:
            break
        offset += ARCGIS_PAGE_SIZE

    if not all_records:
        print("  WARNING: No records returned from ArcGIS.")
        return None

    df = pd.DataFrame(all_records)

    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"], unit="ms", errors="coerce")

    df.to_csv(output_path, index=False)

    h = hashlib.sha256()
    with open(output_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)

    print(f"\n  ArcGIS download complete!")
    print(f"  Records: {len(df):,}")
    print(f"  Date range: {df['day'].min()} to {df['day'].max()}")
    print(f"  Unique sites (WBIDName): {df['WBIDName'].nunique()}")
    print(f"  File: {output_path}")
    print(f"  SHA-256: {h.hexdigest()[:16]}")
    print(f"  Note: Covers 2015+ only. Pre-2015 data requires the static R-Shiny CSV.")

    return output_path


def main():
    """Main execution"""
    config = load_config()
    download_oklahoma_chloride(config)
    download_arcgis_volunteer_data(config)
    print("\n✅ Data extraction complete")

if __name__ == "__main__":
    main()
