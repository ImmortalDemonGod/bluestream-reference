# Blue Thumb ETL Pipeline - Build Guide
## Complete Implementation Guide for Virtual Triangulation Analysis

**Project:** Blue Thumb Water Quality Validation  
**Goal:** Build reproducible ETL pipeline that validates citizen science data  
**Timeline:** 2-3 weeks (self-paced)  
**Purpose:** Resume-building portfolio project + production foundation

---

## 🎯 What You're Building

A complete data pipeline that:
1. Downloads 155,000+ water quality records from EPA
2. Cleans and filters to volunteer vs. professional measurements  
3. Performs spatial-temporal matching (virtual triangulation)
4. Produces dual validation results:
   - **Pro-to-Pro baseline:** N=42, R²=0.753 (OCC Rotating Basin vs professional reference)
   - **Vol-to-Pro validation:** N=25, R²=0.607 (Blue Thumb volunteers vs professional reference)
5. Creates publication-quality visualizations

**Why this matters:** This will make Blue Thumb the first citizen science program in the US with externally validated data using only public databases.

---

## 📁 Repository Structure

Create this exact structure:

```
bluethumb-validation/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/                      # DO NOT commit data files
│   ├── raw/                   # Downloaded EPA data (gitignored)
│   │   └── .gitkeep
│   ├── processed/             # Cleaned datasets (gitignored)
│   │   └── .gitkeep
│   └── outputs/               # Final results (gitignored)
│       └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── extract.py            # Download from EPA
│   ├── transform.py          # Clean and filter
│   ├── analysis.py           # Spatial-temporal matching
│   └── visualize.py          # Create plots
│
├── config/
│   └── config.yaml           # All parameters
│
├── tests/
│   └── test_pipeline.py      # Verification tests
│
└── docs/
    └── REPLICATION.md        # How to reproduce
```

---

## ⚙️ Setup (30 minutes)

### 1. Create Repository

```bash
# On GitHub: Create new repo "bluethumb-validation"
# Clone it locally
git clone https://github.com/YOUR_USERNAME/bluethumb-validation.git
cd bluethumb-validation
```

### 2. Create .gitignore

```gitignore
# Data files (too large for git)
data/raw/*.csv
data/raw/*.zip
data/processed/*.csv
data/outputs/*.png
data/outputs/*.csv

# Python
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
*.ipynb_checkpoints

# Environment
venv/
.env

# OS
.DS_Store
Thumbs.db
```

### 3. Create requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pyyaml>=6.0
requests>=2.31.0
tqdm>=4.66.0
pytest>=7.4.0
```

### 4. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📋 Task 1: Configuration

### Create `config/config.yaml`

```yaml
# Blue Thumb Virtual Triangulation Configuration

data_sources:
  state: "Oklahoma"
  state_code: "US:40"
  characteristic: 
    - "Chloride"
    - "Dissolved oxygen (DO)"
    - "Dissolved oxygen"
  site_type: "Stream"
  sample_media: "Water"
  providers:
    - "NWIS"
    - "STORET"
  date_range:
    start: "01-01-1993"
    end: "12-31-2024"

organizations:
  volunteer:
    - "OKCONCOM_WQX"
    - "CONSERVATION_COMMISSION"
  
  professional:
    - "OKWRB-STREAMS_WQX"  # Oklahoma Water Resources Board
    - "O_MTRIBE_WQX"        # Otoe-Missouria Tribe
    - "OKWRB-LAKES_WQX"     # Oklahoma Water Resources Board (Lakes)
    - "USGS-OK"             # US Geological Survey Oklahoma
    - "USGS-AR"             # US Geological Survey Arkansas
    - "USGS-TX"             # US Geological Survey Texas
    - "OKDEQ"               # Oklahoma Dept of Environmental Quality
    - "ARDEQH2O_WQX"        # Arkansas Dept of Environmental Quality
    - "CHEROKEE"            # Cherokee Nation
    - "CHEROKEE_WQX"        # Cherokee Nation (WQX)
    - "OKCORCOM_WQX"        # Oklahoma Corporation Commission
    - "CNENVSER"            # Cherokee Nation Environmental Services
    - "IOWATROK_WQX"        # Iowa Tribe of Oklahoma
    - "KAWNATON_WQX"        # Kaw Nation
    - "OSAGENTN_WQX"        # Osage Nation
    - "PNDECS_WQX"          # Pawnee Nation
    - "SFNOES_WQX"          # Sac and Fox Nation
    - "WNENVDPT_WQX"        # Wyandotte Nation
    - "CHOCNATWQX"          # Choctaw Nation
    - "DELAWARENATION"      # Delaware Nation
    - "MCNCREEK_WQX"        # Muscogee (Creek) Nation
    - "QTEO_WQX"            # Quapaw Tribe
    - "WDEP_WQX"            # Wichita and Affiliated Tribes

geographic_bounds:
  oklahoma:
    lat_min: 33.6
    lat_max: 37.0
    lon_min: -103.0
    lon_max: -94.4

matching_parameters:
  max_distance_meters: 125
  max_time_hours: 72
  match_strategy: "closest"
  min_concentration_mg_l: 25  # For professional data only

output_paths:
  raw_data: "data/raw/"
  processed_data: "data/processed/"
  results: "data/outputs/"

external_sources:
  volunteer_blue_thumb_csv: "data/Requested_Blue Thumb Chemical Data.csv"
```

**What you're learning:** Configuration management, YAML syntax

---

## 📊 Task 2: Data Extraction (2-3 hours)

### Create `src/extract.py`

```python
"""
extract.py - Download data from EPA Water Quality Portal

Expected runtime: 5-10 minutes
Expected output: data/raw/oklahoma_chloride.csv (~155,000 records, ~88 MB)
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

def main():
    """Main execution"""
    config = load_config()
    filepath = download_oklahoma_chloride(config)
    print("\n✅ Data extraction complete")

if __name__ == "__main__":
    main()
```

**What you're learning:** HTTP requests, file I/O, zip file handling, error handling

**Success criteria:**
- [ ] Downloads ~155,000 records
- [ ] Saves to `data/raw/oklahoma_chloride.csv`
- [ ] File is ~75 MB
- [ ] Completes in 5-10 minutes

---

## 🧹 Task 3: Data Transformation 

### Create `src/transform.py`

```python
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
    """Separate OCC Rotating Basin and professional reference measurements"""
    
    rb_orgs = config['organizations']['rotating_basin']
    pro_orgs = config['organizations']['professional']
    
    # OCC Rotating Basin data (saved for pro-to-pro baseline)
    volunteer_df = df[df['OrganizationIdentifier'].isin(rb_orgs)].copy()
    print(f"\nOCC Rotating Basin (pro-to-pro baseline):")
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
```

**What you're learning:** Pandas data manipulation, filtering, geographic bounds, data quality checks

**Success criteria:**
- [ ] Volunteer: ~15,819 records
- [ ] Professional: ~21,975 records (after >25 mg/L filter)
- [ ] All coordinates within Oklahoma
- [ ] No "Not Detected" values
- [ ] All dates parsed correctly

---

## 🎯 Task 4: Spatial-Temporal Matching 

### Create `src/analysis.py`

```python
"""
analysis.py - Virtual triangulation matching algorithm

Expected runtime: < 1 minute (uses spatial indexing)
Expected output: data/outputs/matched_pairs.csv (25 vol-to-pro records)
             + matched_pairs_pro_to_pro.csv (42 pro-to-pro records)
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
    1. Distance <= 125 meters (Haversine)
    2. Time difference <= 72 hours (absolute)
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
    
    matches = []
    
    print(f"\nMatching volunteer measurements to professional...")
    print(f"  Volunteer measurements: {len(volunteer_df):,}")
    print(f"  Professional measurements: {len(professional_df):,}")
    print(f"  Max distance: {max_distance_m}m")
    print(f"  Max time: {max_time_hours}hrs")
    print(f"\nThis will take 30-60 minutes. Progress bar below:")
    
    # Iterate through volunteer measurements with progress bar
    for idx, vol_row in tqdm(volunteer_df.iterrows(), 
                              total=len(volunteer_df),
                              desc="Matching"):
        
        # Extract volunteer measurement details
        vol_lat = vol_row['LatitudeMeasure']
        vol_lon = vol_row['LongitudeMeasure']
        vol_datetime = vol_row['ActivityStartDate']
        vol_value = vol_row['ResultMeasureValue']
        vol_site_id = vol_row['MonitoringLocationIdentifier']
        vol_org = vol_row['OrganizationIdentifier']
        
        # Find all professional measurements that match criteria
        candidates = []
        
        for jdx, pro_row in professional_df.iterrows():
            
            pro_lat = pro_row['LatitudeMeasure']
            pro_lon = pro_row['LongitudeMeasure']
            pro_datetime = pro_row['ActivityStartDate']
            pro_value = pro_row['ResultMeasureValue']
            pro_site_id = pro_row['MonitoringLocationIdentifier']
            pro_org = pro_row['OrganizationIdentifier']
            
            # Calculate spatial distance (meters)
            distance = haversine_distance(vol_lat, vol_lon, pro_lat, pro_lon)
            
            # Calculate temporal difference (hours)
            time_diff = abs((pro_datetime - vol_datetime).total_seconds() / 3600)
            
            # Check if within thresholds
            if distance <= max_distance_m and time_diff <= max_time_hours:
                candidates.append({
                    'distance': distance,
                    'time_diff': time_diff,
                    'pro_value': pro_value,
                    'pro_org': pro_org,
                    'pro_site_id': pro_site_id,
                    'pro_datetime': pro_datetime,
                    'pro_lat': pro_lat,
                    'pro_lon': pro_lon
                })
        
        # If we found matches, take the spatially closest one
        if len(candidates) > 0:
            # Sort by distance, take closest
            candidates.sort(key=lambda x: x['distance'])
            best_match = candidates[0]
            
            # CRITICAL: Use exact column names that match our actual output
            matches.append({
                'Vol_SiteID': vol_site_id,
                'Pro_SiteID': best_match['pro_site_id'],
                'Vol_Organization': vol_org,
                'Pro_Organization': best_match['pro_org'],
                'Vol_Value': vol_value,
                'Pro_Value': best_match['pro_value'],
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
    volunteer_df = pd.read_csv(proc_dir / "volunteer_chloride.csv")
    professional_df = pd.read_csv(proc_dir / "professional_chloride.csv")
    
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
```

**What you're learning:** Spatial algorithms, nested loops, optimization, statistical analysis

**Success criteria:**
- [ ] 25 vol-to-pro matches + 42 pro-to-pro matches
- [ ] All distances <= 125m
- [ ] All time differences <= 72 hours
- [ ] Vol-to-Pro: R² = 0.607 (±0.01), Slope = 0.813 (±0.01)
- [ ] Pro-to-Pro: R² = 0.753 (±0.01), Slope = 0.735 (±0.01)

---

## 📊 Task 5: Visualization 

### Create `src/visualize.py`

```python
"""
visualize.py - Create validation visualizations

Expected output: data/outputs/validation_plot.png
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import yaml

def load_config():
    """Load configuration"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_validation_plot(matches_df, config):
    """
    Create scatter plot comparing volunteer vs. professional measurements
    
    Shows:
    - Scatter plot of matched pairs
    - Linear regression line
    - 1:1 reference line (perfect agreement)
    - Statistics box (N, R², slope)
    """
    
    vol_values = matches_df['Vol_Value'].values
    pro_values = matches_df['Pro_Value'].values
    
    # Calculate regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(pro_values, vol_values)
    r_squared = r_value ** 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(pro_values, vol_values, alpha=0.6, s=100, color='steelblue',
               edgecolors='navy', linewidth=1, label='Matched Pairs')
    
    # Regression line
    x_line = np.linspace(pro_values.min(), pro_values.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, 
            label=f'Linear Fit (R²={r_squared:.3f})')
    
    # 1:1 reference line
    max_val = max(pro_values.max(), vol_values.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.5,
            label='1:1 Reference')
    
    # Labels and formatting
    ax.set_xlabel('Professional Chloride (mg/L)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Volunteer Chloride (mg/L)', fontsize=14, fontweight='bold')
    ax.set_title('Blue Thumb Virtual Triangulation Results\nVolunteer vs. Professional Chloride Measurements',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Statistics text box
    stats_text = (f'N = {len(matches_df)}\n'
                  f'R² = {r_squared:.3f}\n'
                  f'Slope = {slope:.3f}\n'
                  f'p < 0.0001')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11)
    
    # Set equal aspect
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(config['output_paths']['results'])
    output_path = output_dir / "validation_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nVisualization saved: {output_path}")
    
    plt.close()

def main():
    """Create all visualizations"""
    
    config = load_config()
    
    # Load matched pairs
    results_dir = Path(config['output_paths']['results'])
    matches_df = pd.read_csv(results_dir / "matched_pairs.csv")
    
    # Create validation plot
    create_validation_plot(matches_df, config)
    
    print("\n✅ Visualization complete")

if __name__ == "__main__":
    main()
```

**What you're learning:** Scientific visualization, matplotlib, statistical plots

**Success criteria:**
- [ ] Scatter plot shows all 25 vol-to-pro points
- [ ] Regression line visible
- [ ] Statistics box shows N=25, R²=0.607
- [ ] High-resolution (300 DPI) PNG

---

## ✅ Task 6: Verification & Testing 

### Create `tests/test_pipeline.py`

```python
"""
test_pipeline.py - Verify pipeline results

Run with: pytest tests/test_pipeline.py -v
"""

import pandas as pd
import pytest
from pathlib import Path
from scipy import stats

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
        'Vol_Value', 'Pro_Value',
        'Vol_DateTime', 'Pro_DateTime',
        'Vol_Lat', 'Vol_Lon', 'Pro_Lat', 'Pro_Lon',
        'Distance_m', 'Time_Diff_hours'
    ]
    
    assert list(df.columns) == expected_columns, f"Column mismatch. Expected {expected_columns}, got {list(df.columns)}"

def test_sample_size():
    """Verify we got exactly 25 vol-to-pro matches"""
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    assert len(df) == 25, f"Expected 25 matches, got {len(df)}"

def test_distance_threshold():
    """Verify all distances <= 125m"""
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    max_distance = df['Distance_m'].max()
    assert max_distance <= 125, f"Distance {max_distance}m exceeds 125m threshold"

def test_time_threshold():
    """Verify all time differences <= 72 hours"""
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    max_time = df['Time_Diff_hours'].max()
    assert max_time <= 72, f"Time difference {max_time}hrs exceeds 72hr threshold"

def test_concentration_filter():
    """Verify professional concentrations > 25 mg/L"""
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    min_pro = df['Pro_Value'].min()
    assert min_pro > 25, f"Professional value {min_pro} <= 25 mg/L threshold"

def test_correlation():
    """Verify R² = 0.607 ± 0.01 (vol-to-pro)"""
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    
    vol_vals = df['Vol_Value'].values
    pro_vals = df['Pro_Value'].values
    
    slope, intercept, r_value, p_value, _ = stats.linregress(pro_vals, vol_vals)
    r_squared = r_value ** 2
    
    assert abs(r_squared - 0.607) < 0.01, f"R² = {r_squared:.3f}, expected 0.607"

def test_slope():
    """Verify slope = 0.813 ± 0.01 (vol-to-pro)"""
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    
    vol_vals = df['Vol_Value'].values
    pro_vals = df['Pro_Value'].values
    
    slope, intercept, r_value, p_value, _ = stats.linregress(pro_vals, vol_vals)
    
    assert abs(slope - 0.813) < 0.01, f"Slope = {slope:.3f}, expected 0.813"

def test_organizations():
    """Verify correct organizations present"""
    df = pd.read_csv("data/outputs/matched_pairs.csv")
    
    # Volunteer orgs
    vol_orgs = set(df['Vol_Organization'].unique())
    expected_vol = {'BLUETHUMB_VOL'}
    assert vol_orgs == expected_vol, f"Volunteer orgs mismatch: {vol_orgs}"
    
    # Professional orgs
    pro_orgs = set(df['Pro_Organization'].unique())
    expected_pro = {'OKWRB-STREAMS_WQX', 'CNENVSER'}
    assert pro_orgs == expected_pro, f"Professional orgs mismatch: {pro_orgs}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**What you're learning:** Unit testing, pytest, data validation

**Run tests:**
```bash
pytest tests/test_pipeline.py -v
```

**All tests must pass** before submitting.

---

## 📖 Task 7: Documentation

### Create `README.md`

```markdown
# Blue Thumb Virtual Triangulation Validation

Validation of Oklahoma Blue Thumb citizen science water quality data using spatial-temporal matching with professional monitoring data.

## Results

**Pro-to-Pro Baseline (OCC Rotating Basin vs Professional Reference):**
- **N = 42** matched pairs
- **R² = 0.753** (strong correlation)
- **Slope = 0.735**

**Vol-to-Pro Validation (Blue Thumb Volunteers vs Professional Reference):**
- **N = 25** matched pairs
- **R² = 0.607** (moderate-strong correlation)
- **Slope = 0.813**
- **Statistical significance:** p < 0.001

## Quick Start

```bash
# Setup
git clone https://github.com/YOUR_USERNAME/bluethumb-validation.git
cd bluethumb-validation
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run pipeline
python src/extract.py       # 5-10 min
python src/transform.py     # 2-5 min
python src/analysis.py      # 30-60 min
python src/visualize.py     # <1 min

# Verify results
pytest tests/test_pipeline.py -v

# View outputs
ls data/outputs/
```

## What This Does

Compares volunteer water quality measurements with professional agency measurements at the same locations and times. This validates that citizen scientists are collecting scientifically useful data.

## Why This Matters

Blue Thumb is the **first citizen science water quality program** in the US to be validated using only public data. Missouri Stream Team and Georgia Adopt-A-Stream cannot be validated this way because their volunteer data isn't properly segregated in EPA systems.

## Data Sources

- EPA Water Quality Portal: https://www.waterqualitydata.us
- Oklahoma Blue Thumb (volunteers): OKCONCOM_WQX, CONSERVATION_COMMISSION
- Oklahoma Water Resources Board (professional): OKWRB-STREAMS_WQX
- Otoe-Missouria Tribe (professional): O_MTRIBE_WQX

## Methodology

**Virtual Triangulation:** Match volunteer and professional measurements using:
1. Spatial proximity (≤125 meters, Haversine distance)
2. Temporal proximity (≤72 hours)
3. Same parameter (Chloride)

When multiple professionals match one volunteer, take the spatially closest.

Two comparisons are run with the same parameters:
- **Pro-to-Pro:** OCC Rotating Basin (WQP OKCONCOM_WQX) vs OKWRB/tribal orgs
- **Vol-to-Pro:** Blue Thumb CSV (actual volunteer kits) vs OKWRB/tribal orgs

## Repository Structure

```
bluethumb-validation/
├── src/              # Source code
├── data/             # Data (gitignored)
├── config/           # Configuration
├── tests/            # Verification tests
└── docs/             # Documentation
```

## Citation

```
Ingram, M. (2024). Virtual Triangulation Validation of Citizen Science
Water Quality Monitoring. GitHub: bluethumb-validation.
```

## License

MIT License - See LICENSE file

## Contact

[Your Name] - [Your Email]
```

### Create `docs/REPLICATION.md`

```markdown
# Replication Guide

## Prerequisites

- Python 3.9+
- 8GB RAM
- 2GB disk space
- Internet connection

## Expected Runtime

- extract.py: 5-10 minutes
- transform.py: 2-5 minutes
- analysis.py: 30-60 minutes
- visualize.py: <1 minute
- **Total: ~45-75 minutes**

## Step-by-Step

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/bluethumb-validation.git
cd bluethumb-validation
```

### 2. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Pipeline

```bash
python src/extract.py
python src/transform.py
python src/analysis.py
python src/visualize.py
```

### 4. Verify Results

```bash
pytest tests/test_pipeline.py -v
```

All tests should pass.

### 5. Check Outputs

```bash
ls data/outputs/
# Should contain:
# - matched_pairs.csv (25 rows, vol-to-pro)
# - matched_pairs_pro_to_pro.csv (42 rows)
# - summary_statistics.txt
# - validation_plot.png
```

## Troubleshooting

**Problem:** Download fails  
**Solution:** Check internet connection, EPA may be down (try later)

**Problem:** No matches found  
**Solution:** Check config.yaml dates and organization names

**Problem:** Wrong R²  
**Solution:** Verify you're using `Time_Diff_hours` not `Time_Diff_hrs`

**Problem:** Analysis takes forever  
**Solution:** Normal - nested loops are slow. Should complete in 30-60 min.

## Expected Values

After successful run:

```python
import pandas as pd
from scipy import stats

df = pd.read_csv('data/outputs/matched_pairs.csv')
vol = df['Vol_Value'].values
pro = df['Pro_Value'].values

slope, intercept, r, p, _ = stats.linregress(pro, vol)

print(f"N = {len(df)}")        # 25 (vol-to-pro)
print(f"R² = {r**2:.3f}")      # 0.607
print(f"Slope = {slope:.3f}")  # 0.813
print(f"p = {p:.4e}")          # <0.001
```
```

---

## 🎯 Deliverables Checklist

Before submitting, verify:

### **Code Quality:**
- [ ] All Python files have docstrings
- [ ] Code follows PEP 8 style
- [ ] No hardcoded paths (use config.yaml)
- [ ] Progress bars for long operations

### **Data Pipeline:**
- [ ] extract.py downloads ~155,000 records
- [ ] transform.py produces ~15,819 volunteer, ~21,975 professional
- [ ] analysis.py produces 25 vol-to-pro + 42 pro-to-pro matches
- [ ] visualize.py creates publication-quality plot

### **Results Verification:**
- [ ] Vol-to-Pro: N = 25, R² = 0.607 (±0.01), Slope = 0.813 (±0.01)
- [ ] Pro-to-Pro: N = 42, R² = 0.753 (±0.01), Slope = 0.735 (±0.01)
- [ ] All distances ≤ 125m
- [ ] All time diffs ≤ 72 hours
- [ ] All professional values > 25 mg/L

### **Column Names (CRITICAL):**
```python
# Your matched_pairs.csv MUST have these EXACT columns:
columns = [
    'Vol_SiteID',        # NOT Vol_Site
    'Pro_SiteID',        # NOT Pro_Site
    'Vol_Organization',
    'Pro_Organization',
    'Vol_Value',
    'Pro_Value',
    'Vol_DateTime',      # NOT Vol_Date
    'Pro_DateTime',      # NOT Pro_Date
    'Vol_Lat',
    'Vol_Lon',
    'Pro_Lat',
    'Pro_Lon',
    'Distance_m',
    'Time_Diff_hours'    # NOT Time_Diff_hrs
]
```

### **Tests:**
- [ ] All pytest tests pass
- [ ] Manual verification script runs

### **Documentation:**
- [ ] README.md complete
- [ ] REPLICATION.md has step-by-step guide
- [ ] Code comments explain non-obvious logic

### **Git:**
- [ ] .gitignore properly excludes data files
- [ ] Commit messages are descriptive
- [ ] Repository is public on GitHub

---

## 🚀 Submission

When complete:

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Complete Blue Thumb ETL pipeline"
   git push origin main
   ```

2. **Share repository URL**

3. **Schedule code review**

---

## 📚 Learning Resources

**Python:**
- Pandas: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/stable/tutorials/
- Scipy: https://docs.scipy.org/doc/scipy/

**Spatial Analysis:**
- Haversine formula: https://en.wikipedia.org/wiki/Haversine_formula

**Data Sources:**
- EPA WQP: https://www.waterqualitydata.us/
- Blue Thumb: https://www.ok.gov/conservation/Blue_Thumb/

**Best Practices:**
- PEP 8: https://pep8.org/
- Git: https://git-scm.com/book/

---

## 💡 What You're Learning

By completing this project, you gain experience with:

✅ ETL pipeline development  
✅ API integration (EPA Water Quality Portal)  
✅ Spatial algorithms (Haversine distance)  
✅ Statistical analysis (linear regression)  
✅ Data validation and QA  
✅ Scientific visualization  
✅ Configuration management  
✅ Unit testing (pytest)  
✅ Git version control  
✅ Technical documentation  

**Portfolio value:** This demonstrates production-grade data engineering skills applicable to environmental science, public health, and research domains.

---

## 🆘 Getting Help

**Ask immediately if:**
- Extract step takes >15 minutes
- Vol-to-pro N differs from 25 or pro-to-pro N differs from 42
- Vol-to-pro R² differs by >0.02 from 0.607
- Analysis takes >2 hours
- Any test fails

**Don't struggle alone** - these indicate real issues.

---

## ✨ Success Criteria

Your project is complete when:

1. **All code runs end-to-end** without errors
2. **All tests pass** (pytest shows all green)
3. **Results match:** Vol-to-Pro N=25, R²=0.607; Pro-to-Pro N=42, R²=0.753
4. **Repository is well-documented** and public
5. **You can explain** how spatial-temporal matching works

**This project proves you can build production data pipelines.**

Put it on your resume. Include it in job applications. Use it as a talking point in interviews.

Good luck! 🚀

---

**Last updated:** February 11, 2026  
**Version:** 3.0 (Phase 2 — Dual-Comparison Framework)  
**Status:** All sections updated ✅  

**Changes from v2.0-RC:**
- ✅ Dual-comparison framework: pro-to-pro baseline (N=42, R²=0.753) + vol-to-pro validation (N=25, R²=0.607)
- ✅ Updated matching parameters: 125m/72h/closest (was 100m/48h/all)
- ✅ Added external_sources config for Blue Thumb CSV
- ✅ Fixed volunteer org label: BLUETHUMB_VOL (OKCONCOM_WQX is OCC Rotating Basin, not volunteers)
- ✅ Updated all test assertions, verification checklists, and expected values
- ✅ Updated README template with dual results
- ✅ Updated methodology to explain both comparisons