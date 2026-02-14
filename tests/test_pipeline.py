"""
test_pipeline.py - Verify pipeline results

Run with: pytest tests/test_pipeline.py -v
"""

import pandas as pd
import pytest
from pathlib import Path
from scipy import stats
import yaml


def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_matched_pairs_exist():
    """Verify matched pairs file was created"""
    path = Path("data/outputs/matched_pairs.csv")
    assert path.exists(), "matched_pairs.csv not found"

def test_correct_column_names():
    """Verify exact column names match specification"""
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    
    expected_columns = [
        'Vol_SiteID', 'Pro_SiteID',
        'Vol_Organization', 'Pro_Organization',
        'Pro_Method_ID',
        'Vol_Value', 'Pro_Value',
        'Vol_Units', 'Pro_Units',
        'Vol_DateTime', 'Pro_DateTime',
        'Vol_Lat', 'Vol_Lon', 'Pro_Lat', 'Pro_Lon',
        'Distance_m', 'Time_Diff_hours'
    ]
    
    assert list(df.columns) == expected_columns, f"Column mismatch. Expected {expected_columns}, got {list(df.columns)}"

def test_sample_size():
    """Verify we got at least 20 matches (Phase 2: N=25 expected)"""
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    assert len(df) >= 20, f"Expected at least 20 matches, got {len(df)}"

def test_distance_threshold():
    """Verify all distances within configured threshold"""
    config = load_config()
    max_allowed = config['matching_parameters']['max_distance_meters']
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    if len(df) > 0:
        max_distance = df['Distance_m'].max()
        assert max_distance <= max_allowed, f"Distance {max_distance}m exceeds {max_allowed}m threshold"

def test_time_threshold():
    """Verify all time differences within configured threshold"""
    config = load_config()
    max_allowed = config['matching_parameters']['max_time_hours']
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    if len(df) > 0:
        max_time = df['Time_Diff_hours'].max()
        assert max_time <= max_allowed, f"Time difference {max_time}hrs exceeds {max_allowed}hr threshold"

def test_concentration_filter():
    """Verify professional concentrations respect configured threshold"""
    config = load_config()
    min_conc = config['matching_parameters']['min_concentration_mg_l']
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    if len(df) > 0:
        min_pro = df['Pro_Value'].min()
        assert min_pro > min_conc, f"Professional value {min_pro} <= {min_conc} mg/L threshold"

def test_correlation():
    """Verify R² is reasonable"""
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    
    if len(df) > 2:
        vol_vals = df['Vol_Value'].values
        pro_vals = df['Pro_Value'].values
        
        slope, intercept, r_value, p_value, _ = stats.linregress(pro_vals, vol_vals)
        r_squared = r_value ** 2
        
        assert r_squared > 0.5, f"R² = {r_squared:.3f}, expected > 0.5"

def test_slope():
    """Verify slope is reasonable"""
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    
    if len(df) > 2:
        vol_vals = df['Vol_Value'].values
        pro_vals = df['Pro_Value'].values
        
        slope, intercept, r_value, p_value, _ = stats.linregress(pro_vals, vol_vals)
        
        assert 0.7 < slope < 0.9, f"Slope = {slope:.3f}, expected ~0.81"

def test_organizations():
    """Verify correct organizations present"""
    config = load_config()
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    
    # Volunteer orgs — Blue Thumb data is labeled BLUETHUMB_VOL (not OKCONCOM_WQX,
    # which is OCC Rotating Basin professional data in WQP)
    vol_orgs = set(df['Vol_Organization'].unique())
    expected_vol = {'BLUETHUMB_VOL'}
    assert vol_orgs == expected_vol, f"Expected volunteer orgs {expected_vol}, got {vol_orgs}"
    
    # Professional orgs
    pro_orgs = set(df['Pro_Organization'].unique())
    expected_pro = set(config['organizations']['professional'])
    assert pro_orgs.issubset(expected_pro), f"Unexpected professional orgs: {pro_orgs}"

def test_volunteer_provenance():
    """Verify volunteer data is from Blue Thumb CSV, not raw WQP OKCONCOM_WQX"""
    config = load_config()
    ext_cfg = config.get('external_sources', {})
    ext_path = ext_cfg.get('volunteer_blue_thumb_csv') if isinstance(ext_cfg, dict) else None
    assert ext_path is not None, "No Blue Thumb CSV configured in external_sources"
    assert Path(ext_path).exists(), f"Blue Thumb CSV not found: {ext_path}"
    bt_df = pd.read_csv(ext_path, nrows=5)
    assert 'Chloride' in bt_df.columns or 'WBID' in bt_df.columns, \
        "Blue Thumb CSV missing expected columns (Chloride or WBID)"

def test_spatial_diversity():
    """Verify matches span multiple volunteer sites"""
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    if len(df) > 0:
        unique_vol_sites = df['Vol_SiteID'].nunique()
        assert unique_vol_sites >= 3, f"Only {unique_vol_sites} unique volunteer sites"

def test_professional_diversity():
    """Verify matches include multiple professional organizations"""
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    if len(df) > 0:
        unique_pro_orgs = df['Pro_Organization'].nunique()
        assert unique_pro_orgs >= 2, f"Only {unique_pro_orgs} unique professional orgs"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
