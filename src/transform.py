"""
transform.py - Clean and filter EPA data

Your task: Implement all the cleaning functions
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
    """
    Load raw EPA data
    
    TODO: Implement data loading
    
    Hints:
    - Path is in config['output_paths']['raw_data']
    - Filename is 'oklahoma_chloride.csv'
    - Use low_memory=False to avoid dtype warnings
    """
    # TODO: Build filepath
    # TODO: Load CSV with pandas
    # TODO: Print record count
    # TODO: Return dataframe
    pass

def filter_chloride(df):
    """
    Filter for chloride measurements only
    
    TODO: Implement chloride filter
    
    Hints:
    - Column name: 'CharacteristicName'
    - Keep only rows where CharacteristicName == 'Chloride'
    - Use .copy() to avoid SettingWithCopyWarning
    
    Expected output: ~50,000 records
    """
    # TODO: Filter to Chloride only
    # TODO: Print count
    # TODO: Return filtered dataframe
    pass

def clean_coordinates(df, config):
    """
    Remove invalid coordinates and filter to Oklahoma bounds
    
    TODO: Implement coordinate cleaning
    
    Hints:
    - Columns: 'LatitudeMeasure', 'LongitudeMeasure'
    - First remove null values with .notna()
    - Then filter to Oklahoma bounds from config
    - Bounds are in config['geographic_bounds']['oklahoma']
    
    Expected output: Similar to input (most coordinates are valid)
    """
    # TODO: Get bounds from config
    # TODO: Remove null coordinates
    # TODO: Filter to Oklahoma bounds
    # TODO: Print count after cleaning
    # TODO: Return cleaned dataframe
    pass

def clean_concentrations(df):
    """
    Filter for valid concentration values
    
    TODO: Implement concentration cleaning
    
    Hints:
    - Column: 'ResultMeasureValue'
    - Remove null values
    - Check for 'ResultDetectionConditionText' column
      * If it exists, remove rows where it's not null (these are "Not Detected")
    - Remove negative values
    - Remove extreme outliers (>1000 mg/L)
    
    Expected output: ~45,000 records
    """
    # TODO: Remove null concentrations
    # TODO: Remove "Not Detected" results
    # TODO: Remove negative values
    # TODO: Remove outliers > 1000
    # TODO: Print count after cleaning
    # TODO: Return cleaned dataframe
    pass

def parse_dates(df):
    """
    Convert dates to datetime objects
    
    TODO: Implement date parsing
    
    Hints:
    - Column: 'ActivityStartDate'
    - Use pd.to_datetime() with errors='coerce'
    - Remove rows where date parsing failed (null after conversion)
    
    Expected output: ~44,000 records (very few fail)
    """
    # TODO: Parse dates
    # TODO: Remove null dates
    # TODO: Print count
    # TODO: Return dataframe
    pass

def separate_volunteer_professional(df, config):
    """
    Separate volunteer and professional measurements
    
    TODO: Implement organization separation
    
    Hints:
    - Column: 'OrganizationIdentifier'
    - Volunteer orgs from config['organizations']['volunteer']
    - Professional orgs from config['organizations']['professional']
    - Use .isin() to filter
    - Apply >25 mg/L filter to PROFESSIONAL data only
      * This is in config['matching_parameters']['min_concentration_mg_l']
    
    Expected output:
    - Volunteer: ~15,819 records
    - Professional: ~21,975 records (after concentration filter)
    """
    # TODO: Get organization lists from config
    # TODO: Filter volunteer data
    # TODO: Filter professional data
    # TODO: Apply concentration filter to professional ONLY
    # TODO: Print record counts
    # TODO: Return both dataframes
    pass

def save_processed_data(volunteer_df, professional_df, config):
    """
    Save processed datasets
    
    TODO: Implement saving
    
    Hints:
    - Output directory from config
    - Filenames: 'volunteer_chloride.csv', 'professional_chloride.csv'
    - Use .to_csv() with index=False
    """
    # TODO: Create output directory
    # TODO: Save volunteer data
    # TODO: Save professional data
    # TODO: Print saved file paths
    pass

def main():
    """Main data cleaning pipeline"""
    config = load_config()
    
    # TODO: Call each function in sequence
    # 1. Load raw data
    # 2. Filter to chloride
    # 3. Clean coordinates
    # 4. Clean concentrations
    # 5. Parse dates
    # 6. Separate volunteer/professional
    # 7. Save processed data
    
    print("\n✅ Data transformation complete")

if __name__ == "__main__":
    main()
