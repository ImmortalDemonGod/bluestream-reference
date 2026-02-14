"""
transform.py - Clean and filter EPA data

Expected runtime: 2-5 minutes
Expected output:
  - data/processed/volunteer_chloride.csv (~15,600 records)
  - data/processed/professional_chloride.csv (~18,200 records)
"""

import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def load_config():
    """Load configuration"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_raw_data(config):
    """Load raw EPA data"""
    filepath = Path(config['output_paths']['raw_data']) / "oklahoma_chloride.csv"
    print(f"Loading raw data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    
    # Rename coordinate columns if they exist with prefix
    df.rename(columns={
        'ActivityLocation/LatitudeMeasure': 'LatitudeMeasure',
        'ActivityLocation/LongitudeMeasure': 'LongitudeMeasure'
    }, inplace=True)
    
    print(f"  Loaded {len(df):,} records")
    return df

def load_volunteer_blue_thumb_csv(csv_path: Path) -> pd.DataFrame:
    """Load and normalize Blue Thumb volunteer chemistry CSV to WQP-like schema.

    Expected input columns (subset):
      - 'WBID' (used as MonitoringLocationIdentifier)
      - 'Date' (YYYY-MM-DD)
      - 'Time' (various formats, may be missing)
      - 'Latitude', 'Longitude'
      - 'Chloride' (mg/L)

    Returns a DataFrame with columns required by analysis:
      MonitoringLocationIdentifier, OrganizationIdentifier, LatitudeMeasure,
      LongitudeMeasure, ActivityStartDate, ResultMeasureValue,
      ResultMeasure/MeasureUnitCode
    """
    print(f"\nLoading Blue Thumb volunteer CSV: {csv_path}")

    # Integrity check: SHA-256 hash and file size
    h = hashlib.sha256()
    with open(csv_path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    file_hash = h.hexdigest()[:16]
    file_size = csv_path.stat().st_size
    print(f"  File hash (SHA-256 prefix): {file_hash}")
    print(f"  File size: {file_size / 1024 / 1024:.1f} MB")

    vdf = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw rows: {len(vdf):,}")

    # Rename spatial and ID columns
    vdf = vdf.rename(columns={
        'WBID': 'MonitoringLocationIdentifier',
        'Latitude': 'LatitudeMeasure',
        'Longitude': 'LongitudeMeasure',
    })

    # Parse Date and Time into ActivityStartDate
    # Normalize time strings; fill missing with noon
    raw_time = vdf.get('Time')
    if raw_time is not None:
        time_str = raw_time.astype(str).str.strip()
        # Replace empty or placeholder values
        time_str = time_str.replace({'': '12:00 PM', '.': '12:00 PM', 'nan': '12:00 PM', 'NA': '12:00 PM'})
        # Ensure a space before AM/PM when missing (e.g., '4:00PM' -> '4:00 PM')
        time_str = time_str.str.replace(r'(?i)(am|pm)$', lambda m: ' ' + m.group(1).upper(), regex=True)
    else:
        time_str = '12:00 PM'

    vdf['Date'] = pd.to_datetime(vdf['Date'], errors='coerce')
    # Attempt robust parsing with pandas first
    vdf['ActivityStartDate'] = pd.to_datetime(
        vdf['Date'].astype(str) + ' ' + pd.Series(time_str).astype(str), errors='coerce'
    )
    # Fallback to date-only when time parsing fails
    vdf['ActivityStartDate'] = vdf['ActivityStartDate'].fillna(vdf['Date'])

    # Chloride concentration as numeric
    vdf['ResultMeasureValue'] = pd.to_numeric(vdf.get('Chloride'), errors='coerce')
    vdf = vdf[vdf['ResultMeasureValue'].notna()].copy()
    vdf = vdf[vdf['ResultMeasureValue'] >= 0].copy()

    # Coerce coordinates to numeric and drop invalids
    vdf['LatitudeMeasure'] = pd.to_numeric(vdf['LatitudeMeasure'], errors='coerce')
    vdf['LongitudeMeasure'] = pd.to_numeric(vdf['LongitudeMeasure'], errors='coerce')
    vdf = vdf[vdf['LatitudeMeasure'].notna() & vdf['LongitudeMeasure'].notna()].copy()

    # Set units and organization identifier
    vdf['ResultMeasure/MeasureUnitCode'] = 'mg/L'
    vdf['OrganizationIdentifier'] = 'BLUETHUMB_VOL'

    # Keep only required columns
    required_cols = [
        'MonitoringLocationIdentifier',
        'OrganizationIdentifier',
        'LatitudeMeasure',
        'LongitudeMeasure',
        'ActivityStartDate',
        'ResultMeasureValue',
        'ResultMeasure/MeasureUnitCode',
    ]
    missing = [c for c in required_cols if c not in vdf.columns]
    if missing:
        raise ValueError(f"Blue Thumb CSV missing required columns: {missing}")

    vdf = vdf[required_cols].copy()

    print(f"  Blue Thumb volunteer rows: {len(vdf):,}")
    return vdf

def filter_chloride(df):
    """Filter for chloride measurements only"""
    df = df[df['CharacteristicName'] == 'Chloride'].copy()
    print(f"  Chloride records: {len(df):,}")
    return df

def clean_coordinates(df, config):
    """Remove invalid coordinates"""
    # bounds = config['geographic_bounds']['oklahoma']
    
    # Remove missing coordinates
    df = df[df['LatitudeMeasure'].notna() & df['LongitudeMeasure'].notna()].copy()
    
    # NOTE: Reference implementation does not filter by strict bounds, 
    # relying instead on the state code filter during extraction.
    # Filter to Oklahoma bounds
    # df = df[
    #     (df['LatitudeMeasure'] >= bounds['lat_min']) &
    #     (df['LatitudeMeasure'] <= bounds['lat_max']) &
    #     (df['LongitudeMeasure'] >= bounds['lon_min']) &
    #     (df['LongitudeMeasure'] <= bounds['lon_max'])
    # ].copy()
    
    print(f"  After coordinate cleaning: {len(df):,}")
    return df

def clean_concentrations(df):
    """Filter for valid concentration values"""
    
    # Coerce to numeric, turning non-numerics into NaN
    df['ResultMeasureValue'] = pd.to_numeric(df['ResultMeasureValue'], errors='coerce')
    
    # Remove null values
    df = df[df['ResultMeasureValue'].notna()].copy()
    
    # Remove "Not Detected" results
    if 'ResultDetectionConditionText' in df.columns:
        df = df[df['ResultDetectionConditionText'].isna()].copy()
    
    # Remove negative values
    df = df[df['ResultMeasureValue'] >= 0].copy()
    
    # NOTE: Reference data contains values > 1000 mg/L (max ~1880), so we remove this filter
    # Remove extreme outliers (>1000 mg/L unlikely for chloride)
    # df = df[df['ResultMeasureValue'] < 1000].copy()
    
    print(f"  After concentration cleaning: {len(df):,}")
    return df

def parse_dates(df):
    """Convert dates to datetime objects"""
    df['ActivityStartDate'] = pd.to_datetime(df['ActivityStartDate'], errors='coerce')
    df = df[df['ActivityStartDate'].notna()].copy()
    print(f"  After date parsing: {len(df):,}")
    return df

def separate_volunteer_professional(df, config):
    """Separate OCC Rotating Basin and professional reference measurements.

    Despite the function name, the 'volunteer' split is actually OCC Rotating Basin
    (professional Method 9056). Actual volunteer data is loaded later from the
    external Blue Thumb CSV in process_data().
    """
    rb_orgs = config['organizations']['rotating_basin']
    pro_orgs = config['organizations']['professional']
    
    # OCC Rotating Basin data (labeled "volunteer" historically, but is professional)
    volunteer_df = df[df['OrganizationIdentifier'].isin(rb_orgs)].copy()
    print(f"\nOCC Rotating Basin (will be saved for pro-to-pro baseline):")
    print(f"  Organizations: {rb_orgs}")
    print(f"  Records: {len(volunteer_df):,}")
    
    # Professional data
    professional_df = df[df['OrganizationIdentifier'].isin(pro_orgs)].copy()
    
    # Apply minimum concentration filter to professional data ONLY
    min_conc = config['matching_parameters']['min_concentration_mg_l']
    professional_df = professional_df[
        professional_df['ResultMeasureValue'] > min_conc
    ].copy()
    
    print(f"\nProfessional data:")
    print(f"  Organizations: {pro_orgs}")
    print(f"  Records (before filter): {len(df[df['OrganizationIdentifier'].isin(pro_orgs)]):,}")
    print(f"  Records (after >{min_conc} mg/L filter): {len(professional_df):,}")
    
    return volunteer_df, professional_df

def save_processed_data(volunteer_df, professional_df, config):
    """Save processed datasets"""
    output_dir = Path(config['output_paths']['processed_data'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vol_path = output_dir / "volunteer_chloride.csv"
    pro_path = output_dir / "professional_chloride.csv"
    
    volunteer_df.to_csv(vol_path, index=False)
    professional_df.to_csv(pro_path, index=False)
    
    print(f"\nSaved processed data:")
    print(f"  {vol_path}")
    print(f"  {pro_path}")

def process_data(config):
    """
    Full transformation pipeline: Load, Clean, Separate, Override, Save rotating basin.

    This is the single authoritative entry point for all data processing.
    Both main() and pipeline.py must call this function — never duplicate the logic.

    Returns:
        (volunteer_df, professional_df) ready for matching
    """
    # 1. Load and clean standard WQP data
    df = load_raw_data(config)
    df = filter_chloride(df)
    df = clean_coordinates(df, config)
    df = clean_concentrations(df)
    df = parse_dates(df)

    # 2. Separate by organization
    volunteer_df, professional_df = separate_volunteer_professional(df, config)

    # 3. The WQP OKCONCOM_WQX / CONSERVATION_COMMISSION data is OCC Rotating Basin
    #    (professional lab Method 9056), NOT Blue Thumb volunteers.
    #    Save it separately for the pro-to-pro comparison.
    rotating_basin_df = volunteer_df.copy()
    rb_path = Path(config['output_paths']['processed_data']) / "rotating_basin_chloride.csv"
    rotating_basin_df.to_csv(rb_path, index=False)
    print(f"\nSaved OCC Rotating Basin (pro-to-pro baseline): {rb_path}")
    print(f"  Records: {len(rotating_basin_df):,}")

    # 4. CRITICAL: Override with Blue Thumb CSV (Phase 2 requirement)
    #    The WQP OKCONCOM_WQX data is OCC Rotating Basin (professional), NOT volunteers.
    #    We MUST load the external Blue Thumb CSV for volunteer validation.
    #    If the config key is missing or the file doesn't exist, raise immediately —
    #    never silently fall back to WQP data for volunteers.
    ext_cfg = config.get('external_sources', {})
    ext_path = ext_cfg.get('volunteer_blue_thumb_csv') if isinstance(ext_cfg, dict) else None
    if not ext_path:
        raise ValueError(
            "FATAL: 'external_sources.volunteer_blue_thumb_csv' not set in config.yaml.\n"
            "Phase 2 validation REQUIRES the Blue Thumb CSV. The WQP OKCONCOM_WQX data\n"
            "contains OCC Rotating Basin professionals, not Blue Thumb volunteers."
        )
    vpath = Path(ext_path)
    if not vpath.exists():
        raise FileNotFoundError(
            f"FATAL: Blue Thumb CSV not found at '{vpath}'.\n"
            f"This file is REQUIRED for volunteer validation. The WQP OKCONCOM_WQX data\n"
            f"contains OCC Rotating Basin professionals, not Blue Thumb volunteers.\n"
            f"Download it from OCC R-Shiny: https://occwaterquality.shinyapps.io/OCC-app23a/\n"
            f"Or request a direct export from OCC."
        )
    print("\nLoading Blue Thumb volunteer CSV (actual volunteers)...")
    # Verify file integrity via SHA-256 hash
    h = hashlib.sha256()
    with open(vpath, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    csv_hash = h.hexdigest()
    expected_hash = ext_cfg.get('volunteer_blue_thumb_csv_sha256')
    if expected_hash:
        if csv_hash != expected_hash:
            raise ValueError(
                f"FATAL: Blue Thumb CSV hash mismatch!\n"
                f"  Expected: {expected_hash}\n"
                f"  Got:      {csv_hash}\n"
                f"The file may have been modified or replaced. Verify provenance."
            )
        print(f"  SHA-256 verified: {csv_hash[:16]}...")
    else:
        print(f"  SHA-256: {csv_hash}")
        print(f"  (Add 'volunteer_blue_thumb_csv_sha256: {csv_hash}' to config.yaml to enforce)")
    volunteer_df = load_volunteer_blue_thumb_csv(vpath)

    return volunteer_df, professional_df


def main():
    """Main data cleaning pipeline"""
    config = load_config()
    volunteer_df, professional_df = process_data(config)
    save_processed_data(volunteer_df, professional_df, config)
    print("\n✅ Data transformation complete")

if __name__ == "__main__":
    main()
