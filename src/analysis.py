"""
analysis.py - Virtual triangulation matching algorithm

Your task: Implement the Haversine distance and matching algorithm

This is the core of the project - take your time!
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from tqdm import tqdm
import yaml

def load_config():
    """Load configuration"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points on Earth
    
    TODO: Implement Haversine formula
    
    The Haversine formula:
    a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    c = 2 × atan2(√a, √(1−a))
    distance = R × c
    
    Where:
    - R = Earth's radius = 6,371,000 meters
    - Δlat = lat2 - lat1 (in radians)
    - Δlon = lon2 - lon1 (in radians)
    
    Args:
        lat1, lon1: First point (decimal degrees)
        lat2, lon2: Second point (decimal degrees)
        
    Returns:
        Distance in meters
        
    Test case:
        Oklahoma City to Tulsa should be ~160 km
        OKC: (35.4676, -97.5164)
        Tulsa: (36.1540, -95.9928)
    """
    
    # TODO: Define Earth's radius in meters
    R = ... # TODO
    
    # TODO: Convert decimal degrees to radians
    # HINT: Use np.radians()
    
    # TODO: Calculate differences
    dlat = ... # TODO
    dlon = ... # TODO
    
    # TODO: Implement Haversine formula
    # HINT: Use np.sin(), np.cos(), np.arctan2(), np.sqrt()
    
    # TODO: Return distance in meters
    pass

def find_matches(volunteer_df, professional_df, config):
    """
    Find volunteer-professional measurement pairs
    
    TODO: Implement spatial-temporal matching
    
    Algorithm:
    1. For each volunteer measurement:
       a. Loop through ALL professional measurements
       b. Calculate spatial distance (use haversine_distance)
       c. Calculate temporal difference in hours
       d. If BOTH within thresholds, add to candidates list
       e. If multiple candidates, take the closest in SPACE
       f. Record the match
    
    CRITICAL - Your output DataFrame MUST have these EXACT column names:
    - Vol_SiteID (NOT Vol_Site)
    - Pro_SiteID (NOT Pro_Site)  
    - Vol_Organization
    - Pro_Organization
    - Vol_Value
    - Pro_Value
    - Vol_DateTime (NOT Vol_Date)
    - Pro_DateTime (NOT Pro_Date)
    - Vol_Lat
    - Vol_Lon
    - Pro_Lat
    - Pro_Lon
    - Distance_m
    - Time_Diff_hours (NOT Time_Diff_hrs)
    
    Args:
        volunteer_df: Volunteer measurements
        professional_df: Professional measurements
        config: Configuration dictionary
        
    Returns:
        DataFrame with matched pairs
        
    Expected output: 48 matches
    """
    
    # TODO: Get thresholds from config
    max_distance_m = ... # TODO
    max_time_hours = ... # TODO
    
    matches = []
    
    print(f"\nMatching volunteer measurements to professional...")
    print(f"  Volunteer measurements: {len(volunteer_df):,}")
    print(f"  Professional measurements: {len(professional_df):,}")
    print(f"  Max distance: {max_distance_m}m")
    print(f"  Max time: {max_time_hours}hrs")
    print(f"\nThis will take 30-60 minutes. Progress bar below:")
    
    # TODO: Add progress bar with tqdm
    # HINT: for idx, vol_row in tqdm(volunteer_df.iterrows(), total=len(volunteer_df)):
    
    for idx, vol_row in volunteer_df.iterrows():
        
        # TODO: Extract volunteer measurement details
        vol_lat = ... # TODO: Get 'LatitudeMeasure'
        vol_lon = ... # TODO: Get 'LongitudeMeasure'
        vol_datetime = ... # TODO: Get 'ActivityStartDate'
        vol_value = ... # TODO: Get 'ResultMeasureValue'
        vol_site_id = ... # TODO: Get 'MonitoringLocationIdentifier'
        vol_org = ... # TODO: Get 'OrganizationIdentifier'
        
        # TODO: Find all professional measurements that match
        candidates = []
        
        for jdx, pro_row in professional_df.iterrows():
            
            # TODO: Extract professional measurement details
            pro_lat = ... # TODO
            pro_lon = ... # TODO
            pro_datetime = ... # TODO
            pro_value = ... # TODO
            pro_site_id = ... # TODO
            pro_org = ... # TODO
            
            # TODO: Calculate spatial distance in meters
            distance = ... # TODO: Call haversine_distance()
            
            # TODO: Calculate temporal difference in hours
            # HINT: time_diff = abs((pro_datetime - vol_datetime).total_seconds() / 3600)
            time_diff = ... # TODO
            
            # TODO: Check if within thresholds
            if distance <= max_distance_m and time_diff <= max_time_hours:
                # TODO: Add to candidates list
                # HINT: Store all the info you'll need later
                candidates.append({
                    'distance': distance,
                    'time_diff': time_diff,
                    # TODO: Add other fields
                })
        
        # TODO: If we found matches, take the spatially closest one
        if len(candidates) > 0:
            # TODO: Sort candidates by distance
            # HINT: candidates.sort(key=lambda x: x['distance'])
            
            # TODO: Take the first (closest) match
            best_match = ... # TODO
            
            # TODO: Create match record with EXACT column names
            matches.append({
                'Vol_SiteID': vol_site_id,  # NOT Vol_Site!
                'Pro_SiteID': best_match['pro_site_id'],  # NOT Pro_Site!
                # TODO: Fill in all other fields
                # REMEMBER: Vol_DateTime not Vol_Date
                # REMEMBER: Time_Diff_hours not Time_Diff_hrs
            })
    
    return pd.DataFrame(matches)

def calculate_statistics(matches_df):
    """
    Calculate correlation and regression statistics
    
    TODO: Implement statistical analysis
    
    Hints:
    - Use scipy.stats.linregress()
    - X-axis: Professional values (Pro_Value)
    - Y-axis: Volunteer values (Vol_Value)
    - Calculate R² from r_value
    
    Expected results:
    - N = 48
    - R² ≈ 0.839
    - Slope ≈ 0.712
    - p-value < 0.0001
    """
    
    # TODO: Extract values
    vol_values = ... # TODO
    pro_values = ... # TODO
    
    # TODO: Run linear regression
    # HINT: slope, intercept, r_value, p_value, std_err = stats.linregress(pro_values, vol_values)
    
    # TODO: Calculate R²
    
    # TODO: Return statistics dictionary
    return {
        'n': ..., # TODO,
        'r_squared': ..., # TODO,
        'slope': ..., # TODO,
        'intercept': ..., # TODO,
        'p_value': ... # TODO
    }

def save_results(matches_df, stats, config):
    """
    Save matched pairs and statistics
    
    TODO: Implement saving
    
    Hints:
    - Create output directory
    - Save matched_pairs.csv
    - Save summary_statistics.txt with results
    """
    # TODO: Implement
    pass

def main():
    """Run virtual triangulation analysis"""
    
    config = load_config()
    
    # TODO: Load processed data from data/processed/
    # TODO: Parse ActivityStartDate to datetime
    
    # TODO: Find matches
    
    # TODO: Calculate statistics
    
    # TODO: Display results
    
    # TODO: Save results
    
    print("\n✅ Virtual triangulation analysis complete")

if __name__ == "__main__":
    main()
