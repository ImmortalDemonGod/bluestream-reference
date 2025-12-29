"""
analysis.py - Virtual triangulation matching algorithm

Expected runtime: < 1 minute (uses spatial indexing)
Expected output: data/outputs/matched_pairs.csv (48 records)
"""

import pandas as pd
import numpy as np
from scipy import stats, spatial
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
    
    Args:
        lat1, lon1: First point (decimal degrees)
        lat2, lon2: Second point (decimal degrees)
        
    Returns:
        Distance in meters
    """
    
    # Earth's radius in meters
    R = 6371000
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    
    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def find_matches(volunteer_df, professional_df, config):
    """
    Find volunteer-professional measurement pairs using spatial-temporal matching
    
    Matching criteria:
    1. Distance <= 100 meters (Haversine)
    2. Time difference <= 48 hours (absolute)
    3. If multiple professionals match one volunteer, take closest in space
    
    Args:
        volunteer_df: Volunteer measurements
        professional_df: Professional measurements
        config: Configuration dictionary
        
    Returns:
        DataFrame with matched pairs (EXACT column names matter!)
    """
    
    params = config['matching_parameters']
    max_distance_m = params['max_distance_meters']
    max_time_hours = params['max_time_hours']
    match_strategy = params['match_strategy']
    
    matches = []
    
    print(f"\nMatching volunteer measurements to professional...")
    print(f"  Volunteer measurements: {len(volunteer_df):,}")
    print(f"  Professional measurements: {len(professional_df):,}")
    print(f"  Max distance: {max_distance_m}m")
    print(f"  Max time: {max_time_hours}hrs")
    print(f"\nUsing cKDTree for spatial indexing...")
    
    # Prepare data for cKDTree
    vol_coords = volunteer_df[['LatitudeMeasure', 'LongitudeMeasure']].values
    pro_coords = professional_df[['LatitudeMeasure', 'LongitudeMeasure']].values
    
    # Convert professional dataframe to records for O(1) access
    # This is CRITICAL for performance. iloc inside a loop is too slow.
    pro_records = professional_df.to_dict('records')
    
    # Build Tree on Professional Data
    tree = spatial.cKDTree(pro_coords)
    
    # Query for candidates within ~250 meters (approx 0.0025 degrees) to be safe
    # 1 degree lat ~ 111km. 100m = 0.0009 deg. Using 0.0025 as buffer.
    search_radius_deg = 0.0025
    
    # query_ball_point returns list of indices for each volunteer point
    indices_list = tree.query_ball_point(vol_coords, search_radius_deg)
    
    # Iterate through volunteer measurements
    for idx, pro_indices in tqdm(enumerate(indices_list), 
                                 total=len(volunteer_df), 
                                 desc="Matching"):
        
        if not pro_indices:
            continue
            
        vol_row = volunteer_df.iloc[idx]
        
        # Extract volunteer details
        vol_lat = vol_row['LatitudeMeasure']
        vol_lon = vol_row['LongitudeMeasure']
        vol_datetime = vol_row['ActivityStartDate']
        vol_value = vol_row['ResultMeasureValue']
        vol_site_id = vol_row['MonitoringLocationIdentifier']
        vol_org = vol_row['OrganizationIdentifier']

        vol_units = np.nan
        if 'ResultMeasure/MeasureUnitCode' in volunteer_df.columns:
            vol_units = vol_row['ResultMeasure/MeasureUnitCode']

        if match_strategy == 'all':
            # Append all qualifying matches (reference-style behavior)
            for pro_idx in pro_indices:
                pro_row = pro_records[pro_idx]

                # Calculate EXACT Haversine distance
                distance = haversine_distance(
                    vol_lat,
                    vol_lon,
                    pro_row['LatitudeMeasure'],
                    pro_row['LongitudeMeasure'],
                )

                if distance > max_distance_m:
                    continue

                # Calculate temporal difference
                time_diff = abs((pro_row['ActivityStartDate'] - vol_datetime).total_seconds() / 3600)

                if time_diff > max_time_hours:
                    continue

                pro_units = np.nan
                if 'ResultMeasure/MeasureUnitCode' in professional_df.columns:
                    pro_units = pro_row.get('ResultMeasure/MeasureUnitCode', np.nan)

                matches.append({
                    'Vol_SiteID': vol_site_id,
                    'Pro_SiteID': pro_row['MonitoringLocationIdentifier'],
                    'Vol_Organization': vol_org,
                    'Pro_Organization': pro_row['OrganizationIdentifier'],
                    'Vol_Value': vol_value,
                    'Pro_Value': pro_row['ResultMeasureValue'],
                    'Vol_Units': vol_units,
                    'Pro_Units': pro_units,
                    'Vol_DateTime': vol_datetime,
                    'Pro_DateTime': pro_row['ActivityStartDate'],
                    'Vol_Lat': vol_lat,
                    'Vol_Lon': vol_lon,
                    'Pro_Lat': pro_row['LatitudeMeasure'],
                    'Pro_Lon': pro_row['LongitudeMeasure'],
                    'Distance_m': distance,
                    'Time_Diff_hours': time_diff,
                })

            continue

        candidates = []
        
        # Check specific candidates from spatial index
        for pro_idx in pro_indices:
            pro_row = pro_records[pro_idx]
            
            # Calculate EXACT Haversine distance
            distance = haversine_distance(vol_lat, vol_lon, 
                                        pro_row['LatitudeMeasure'], 
                                        pro_row['LongitudeMeasure'])
            
            if distance > max_distance_m:
                continue
                
            # Calculate temporal difference
            time_diff = abs((pro_row['ActivityStartDate'] - vol_datetime).total_seconds() / 3600)
            
            if time_diff <= max_time_hours:
                pro_units = np.nan
                if 'ResultMeasure/MeasureUnitCode' in professional_df.columns:
                    pro_units = pro_row.get('ResultMeasure/MeasureUnitCode', np.nan)
                candidates.append({
                    'distance': distance,
                    'time_diff': time_diff,
                    'pro_value': pro_row['ResultMeasureValue'],
                    'pro_units': pro_units,
                    'pro_org': pro_row['OrganizationIdentifier'],
                    'pro_site_id': pro_row['MonitoringLocationIdentifier'],
                    'pro_datetime': pro_row['ActivityStartDate'],
                    'pro_lat': pro_row['LatitudeMeasure'],
                    'pro_lon': pro_row['LongitudeMeasure']
                })
        
        # If we found matches, take the spatially closest one
        if len(candidates) > 0:
            candidates.sort(key=lambda x: x['distance'])
            best_match = candidates[0]
            
            matches.append({
                'Vol_SiteID': vol_site_id,
                'Pro_SiteID': best_match['pro_site_id'],
                'Vol_Organization': vol_org,
                'Pro_Organization': best_match['pro_org'],
                'Vol_Value': vol_value,
                'Pro_Value': best_match['pro_value'],
                'Vol_Units': vol_units,
                'Pro_Units': best_match['pro_units'],
                'Vol_DateTime': vol_datetime,
                'Pro_DateTime': best_match['pro_datetime'],
                'Vol_Lat': vol_lat,
                'Vol_Lon': vol_lon,
                'Pro_Lat': best_match['pro_lat'],
                'Pro_Lon': best_match['pro_lon'],
                'Distance_m': best_match['distance'],
                'Time_Diff_hours': best_match['time_diff']
            })
    
    return pd.DataFrame(matches)

def calculate_statistics(matches_df):
    """Calculate correlation and regression statistics"""
    
    vol_values = matches_df['Vol_Value'].values
    pro_values = matches_df['Pro_Value'].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(pro_values, vol_values)
    r_squared = r_value ** 2
    
    return {
        'n': len(matches_df),
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'p_value': p_value
    }

def save_results(matches_df, stats, config):
    """Save matched pairs and statistics"""
    
    output_dir = Path(config['output_paths']['results'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save matched pairs
    matches_path = output_dir / "matched_pairs.csv"
    matches_df.to_csv(matches_path, index=False)
    
    # Save statistics
    stats_path = output_dir / "summary_statistics.txt"
    with open(stats_path, 'w') as f:
        f.write("Blue Thumb Virtual Triangulation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Sample Size: N = {stats['n']}\n")
        f.write(f"Correlation: R² = {stats['r_squared']:.3f}\n")
        f.write(f"Slope: {stats['slope']:.3f}\n")
        f.write(f"Intercept: {stats['intercept']:.3f}\n")
        f.write(f"P-value: {stats['p_value']:.4e}\n")
    
    print(f"\nResults saved:")
    print(f"  {matches_path}")
    print(f"  {stats_path}")
    
    return matches_path, stats_path

def main():
    """Run virtual triangulation analysis"""
    
    config = load_config()
    
    # Load processed data
    proc_dir = Path(config['output_paths']['processed_data'])
    volunteer_df = pd.read_csv(proc_dir / "volunteer_chloride.csv", low_memory=False)
    professional_df = pd.read_csv(proc_dir / "professional_chloride.csv", low_memory=False)
    
    # Parse dates
    volunteer_df['ActivityStartDate'] = pd.to_datetime(volunteer_df['ActivityStartDate'])
    professional_df['ActivityStartDate'] = pd.to_datetime(professional_df['ActivityStartDate'])
    
    # Find matches
    matches_df = find_matches(volunteer_df, professional_df, config)
    
    # Calculate statistics
    stats = calculate_statistics(matches_df)
    
    # Display results
    print(f"\n{'='*50}")
    print(f"VIRTUAL TRIANGULATION RESULTS")
    print(f"{'='*50}")
    print(f"\nSample Size: N = {stats['n']}")
    print(f"Correlation: R² = {stats['r_squared']:.3f}")
    print(f"Slope: {stats['slope']:.3f}")
    print(f"P-value: {stats['p_value']:.4e}")
    
    # Save results
    save_results(matches_df, stats, config)
    
    print("\n✅ Virtual triangulation analysis complete")

if __name__ == "__main__":
    main()
