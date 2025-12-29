"""
transform.py - Clean and filter EPA data

Expected runtime: 2-5 minutes
Expected output:
  - data/processed/volunteer_chloride.csv (~15,600 records)
  - data/processed/professional_chloride.csv (~18,200 records)
"""

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
    """Separate volunteer and professional measurements"""
    
    vol_orgs = config['organizations']['volunteer']
    pro_orgs = config['organizations']['professional']
    
    # Volunteer data
    volunteer_df = df[df['OrganizationIdentifier'].isin(vol_orgs)].copy()
    print(f"\nVolunteer data:")
    print(f"  Organizations: {vol_orgs}")
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

def main():
    """Main data cleaning pipeline"""
    config = load_config()
    
    # Load and clean
    df = load_raw_data(config)
    df = filter_chloride(df)
    df = clean_coordinates(df, config)
    df = clean_concentrations(df)
    df = parse_dates(df)
    
    # Separate volunteer/professional
    volunteer_df, professional_df = separate_volunteer_professional(df, config)
    
    # Save
    save_processed_data(volunteer_df, professional_df, config)
    
    print("\n✅ Data transformation complete")

if __name__ == "__main__":
    main()
