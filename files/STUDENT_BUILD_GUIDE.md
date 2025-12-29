# Blue Thumb ETL Pipeline - Student Build Guide
## Learn Data Engineering by Building a Real Validation System

**Project:** Blue Thumb Water Quality Validation  
**Goal:** Build your own ETL pipeline from scratch (with guidance)
**Timeline:** 2-3 weeks (self-paced)  
**Purpose:** Learn by doing + create portfolio piece you can actually explain

---

## 🎯 What You're Building

A complete data pipeline that:
1. Downloads 155,000+ water quality records from EPA
2. Cleans and filters to volunteer vs. professional measurements  
3. Performs spatial-temporal matching (virtual triangulation)
4. Produces validation results: **N=48 matches, R²=0.839**
5. Creates publication-quality visualizations

**Why this matters:** You'll learn production ETL skills while validating citizen science data.

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

# Create the directory structure
mkdir -p data/raw data/processed data/outputs
mkdir -p src config tests docs
touch data/raw/.gitkeep data/processed/.gitkeep data/outputs/.gitkeep
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

## 📋 Task 1: Configuration (30 minutes)

**Goal:** Create configuration file with all pipeline parameters.

### Create `config/config.yaml`

```yaml
# Blue Thumb Virtual Triangulation Configuration

data_sources:
  state: "Oklahoma"
  state_code: "US:40"
  characteristic: "Chloride"
  date_range:
    start: "1993-01-01"
    end: "2024-12-31"

organizations:
  volunteer:
    - "OKCONCOM_WQX"
    - "CONSERVATION_COMMISSION"
  
  professional:
    - "OKWRB-STREAMS_WQX"  # Oklahoma Water Resources Board
    - "O_MTRIBE_WQX"        # Otoe-Missouria Tribe
    # IMPORTANT: NO USGS-OK (zero matches exist in data)

geographic_bounds:
  oklahoma:
    lat_min: 33.6
    lat_max: 37.0
    lon_min: -103.0
    lon_max: -94.4

matching_parameters:
  max_distance_meters: 100
  max_time_hours: 48
  min_concentration_mg_l: 25  # For professional data only

output_paths:
  raw_data: "data/raw/"
  processed_data: "data/processed/"
  results: "data/outputs/"
```

**Verification:** 
- [ ] File created at `config/config.yaml`
- [ ] Valid YAML syntax (no tabs, correct indentation)
- [ ] Can load with: `python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"`

---

## 📊 Task 2: Data Extraction (2-3 hours)

**Goal:** Download chloride data from EPA Water Quality Portal.

**Expected runtime:** 5-10 minutes  
**Expected output:** `data/raw/oklahoma_chloride.csv` (~155,000 records, ~75 MB)

### Create `src/extract.py`

```python
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
```

**Learning Resources:**
- Requests library: https://docs.python-requests.org/
- Zipfile module: https://docs.python.org/3/library/zipfile.html
- Pathlib: https://docs.python.org/3/library/pathlib.html

**Verification Checklist:**
- [ ] File exists at `data/raw/oklahoma_chloride.csv`
- [ ] File size is ~70-80 MB
- [ ] Record count is ~150,000-160,000
- [ ] Can load with pandas: `pd.read_csv('data/raw/oklahoma_chloride.csv')`
- [ ] Script completes in 5-10 minutes

**Common Errors:**
- If download fails: Check internet connection, EPA may be temporarily down
- If extraction fails: Check that zip contains files with 'result' in name
- If too slow: Add progress indicator (optional)

---

## 🧹 Task 3: Data Transformation (3-4 hours)

**Goal:** Clean raw data and separate volunteer from professional measurements.

**Expected runtime:** 2-5 minutes  
**Expected output:**
- `data/processed/volunteer_chloride.csv` (~15,819 records)
- `data/processed/professional_chloride.csv` (~21,975 records after filter)

### Create `src/transform.py`

```python
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
```

**Learning Resources:**
- Pandas filtering: https://pandas.pydata.org/docs/user_guide/indexing.html
- Missing data: https://pandas.pydata.org/docs/user_guide/missing_data.html
- DateTime: https://pandas.pydata.org/docs/user_guide/timeseries.html

**Verification Checklist:**
- [ ] `data/processed/volunteer_chloride.csv` exists
- [ ] Volunteer records: 15,000-16,000
- [ ] `data/processed/professional_chloride.csv` exists
- [ ] Professional records: 21,000-22,000
- [ ] All professional concentrations > 25 mg/L
- [ ] No null coordinates in either file
- [ ] All dates are valid datetime

**Debugging Tips:**
```python
# Check your intermediate steps
df = pd.read_csv('data/raw/oklahoma_chloride.csv', low_memory=False)
print(f"Raw: {len(df)}")

df = df[df['CharacteristicName'] == 'Chloride']
print(f"After chloride filter: {len(df)}")

# Check organizations present
print(df['OrganizationIdentifier'].value_counts())
```

---

## 🎯 Task 4: Spatial-Temporal Matching (4-6 hours)

**Goal:** Implement virtual triangulation algorithm.

**Expected runtime:** 30-60 minutes  
**Expected output:** `data/outputs/matched_pairs.csv` (48 records)

### Create `src/analysis.py`

```python
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
    R = # TODO
    
    # TODO: Convert decimal degrees to radians
    # HINT: Use np.radians()
    
    # TODO: Calculate differences
    dlat = # TODO
    dlon = # TODO
    
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
    max_distance_m = # TODO
    max_time_hours = # TODO
    
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
        vol_lat = # TODO: Get 'LatitudeMeasure'
        vol_lon = # TODO: Get 'LongitudeMeasure'
        vol_datetime = # TODO: Get 'ActivityStartDate'
        vol_value = # TODO: Get 'ResultMeasureValue'
        vol_site_id = # TODO: Get 'MonitoringLocationIdentifier'
        vol_org = # TODO: Get 'OrganizationIdentifier'
        
        # TODO: Find all professional measurements that match
        candidates = []
        
        for jdx, pro_row in professional_df.iterrows():
            
            # TODO: Extract professional measurement details
            pro_lat = # TODO
            pro_lon = # TODO
            pro_datetime = # TODO
            pro_value = # TODO
            pro_site_id = # TODO
            pro_org = # TODO
            
            # TODO: Calculate spatial distance in meters
            distance = # TODO: Call haversine_distance()
            
            # TODO: Calculate temporal difference in hours
            # HINT: time_diff = abs((pro_datetime - vol_datetime).total_seconds() / 3600)
            time_diff = # TODO
            
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
            best_match = # TODO
            
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
    vol_values = # TODO
    pro_values = # TODO
    
    # TODO: Run linear regression
    # HINT: slope, intercept, r_value, p_value, std_err = stats.linregress(pro_values, vol_values)
    
    # TODO: Calculate R²
    
    # TODO: Return statistics dictionary
    return {
        'n': # TODO,
        'r_squared': # TODO,
        'slope': # TODO,
        'intercept': # TODO,
        'p_value': # TODO
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
```

**Learning Resources:**
- Haversine formula: https://en.wikipedia.org/wiki/Haversine_formula
- SciPy stats: https://docs.scipy.org/doc/scipy/reference/stats.html
- tqdm progress bars: https://tqdm.github.io/

**Verification Checklist:**
- [ ] Haversine test: OKC to Tulsa ≈ 160 km
- [ ] Exactly 48 matches found
- [ ] All distances ≤ 100m
- [ ] All time differences ≤ 48 hours
- [ ] R² = 0.839 (±0.001)
- [ ] Slope = 0.712 (±0.001)
- [ ] Column names EXACTLY match specification

**Debugging the Haversine Formula:**
```python
# Test your implementation
from src.analysis import haversine_distance

# Oklahoma City to Tulsa
okc_lat, okc_lon = 35.4676, -97.5164
tul_lat, tul_lon = 36.1540, -95.9928

distance = haversine_distance(okc_lat, okc_lon, tul_lat, tul_lon)
print(f"Distance: {distance/1000:.1f} km")  # Should be ~160 km

# If your answer is way off:
# - Check if you converted degrees to radians
# - Check if you're using the right Earth radius (6371000 m)
# - Check the formula implementation step by step
```

**Performance Note:**
This will be SLOW (~30-60 minutes) because of nested loops. That's okay! You're iterating ~15,000 × ~22,000 = 330 million times. This is a known bottleneck. In production, you'd optimize with spatial indexing, but for learning, the simple approach is fine.

---

## 📊 Task 5: Visualization (2 hours)

**Goal:** Create publication-quality validation plot.

**Expected output:** `data/outputs/validation_plot.png`

### Create `src/visualize.py`

```python
"""
visualize.py - Create validation visualizations

Your task: Create a scatter plot comparing volunteer vs professional
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
    
    TODO: Create publication-quality plot
    
    Your plot should include:
    1. Scatter plot of matched pairs
       - X-axis: Professional chloride (Pro_Value)
       - Y-axis: Volunteer chloride (Vol_Value)
    2. Linear regression line
    3. 1:1 reference line (perfect agreement)
    4. Statistics text box showing N, R², slope
    5. Proper labels and title
    
    Hints:
    - Use plt.subplots(figsize=(10, 8))
    - Use ax.scatter() for points
    - Use stats.linregress() for regression line
    - Use ax.plot() for lines
    - Use ax.text() for statistics box
    - Save with dpi=300 for publication quality
    """
    
    # TODO: Extract values
    vol_values = # TODO
    pro_values = # TODO
    
    # TODO: Calculate regression
    # HINT: slope, intercept, r_value, p_value, std_err = stats.linregress(...)
    
    # TODO: Create figure and axis
    fig, ax = # TODO
    
    # TODO: Create scatter plot
    # HINT: ax.scatter(pro_values, vol_values, ...)
    # Suggestion: Use alpha=0.6, color='steelblue', s=100
    
    # TODO: Add regression line
    # HINT: Create x_line = np.linspace(min, max)
    # HINT: Calculate y_line = slope * x_line + intercept
    # HINT: ax.plot(x_line, y_line, 'r-', ...)
    
    # TODO: Add 1:1 reference line
    # HINT: max_val = max(pro_values.max(), vol_values.max())
    # HINT: ax.plot([0, max_val], [0, max_val], 'k--', ...)
    
    # TODO: Add labels
    # ax.set_xlabel('Professional Chloride (mg/L)', ...)
    # ax.set_ylabel('Volunteer Chloride (mg/L)', ...)
    # ax.set_title(...)
    
    # TODO: Add statistics text box
    # HINT: stats_text = f'N = {len(matches_df)}\nR² = {r_squared:.3f}\n...'
    # HINT: ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, ...)
    
    # TODO: Add grid and legend
    
    # TODO: Save figure
    # HINT: output_path = Path(config['output_paths']['results']) / "validation_plot.png"
    # HINT: plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def main():
    """Create all visualizations"""
    
    config = load_config()
    
    # TODO: Load matched pairs
    
    # TODO: Create validation plot
    
    print("\n✅ Visualization complete")

if __name__ == "__main__":
    main()
```

**Learning Resources:**
- Matplotlib tutorial: https://matplotlib.org/stable/tutorials/
- Scatter plots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
- Text annotations: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html

**Verification Checklist:**
- [ ] File exists: `data/outputs/validation_plot.png`
- [ ] Plot shows 48 points
- [ ] Regression line visible
- [ ] 1:1 reference line visible
- [ ] Statistics box shows: N=48, R²=0.839, Slope=0.712
- [ ] Axes labeled correctly
- [ ] Title present
- [ ] Image is high resolution (300 DPI)

---

## ✅ Task 6: Testing (1 hour)

**Goal:** Write tests to verify your pipeline works correctly.

### Create `tests/test_pipeline.py`

```python
"""
test_pipeline.py - Verify pipeline results

Your task: Implement verification tests

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
    """
    Verify exact column names match specification
    
    TODO: Load the CSV and check column names
    
    Expected columns (EXACT - order matters):
    ['Vol_SiteID', 'Pro_SiteID',
     'Vol_Organization', 'Pro_Organization',
     'Vol_Value', 'Pro_Value',
     'Vol_DateTime', 'Pro_DateTime',
     'Vol_Lat', 'Vol_Lon', 'Pro_Lat', 'Pro_Lon',
     'Distance_m', 'Time_Diff_hours']
    """
    # TODO: Load CSV
    # TODO: Check list(df.columns) == expected_columns
    pass

def test_sample_size():
    """
    Verify we got exactly 48 matches
    
    TODO: Load CSV and check len(df) == 48
    """
    # TODO: Implement
    pass

def test_distance_threshold():
    """
    Verify all distances <= 100m
    
    TODO: Load CSV and check df['Distance_m'].max() <= 100
    """
    # TODO: Implement
    pass

def test_time_threshold():
    """
    Verify all time differences <= 48 hours
    
    TODO: Load CSV and check df['Time_Diff_hours'].max() <= 48
    """
    # TODO: Implement
    pass

def test_concentration_filter():
    """
    Verify professional concentrations > 25 mg/L
    
    TODO: Load CSV and check df['Pro_Value'].min() > 25
    """
    # TODO: Implement
    pass

def test_correlation():
    """
    Verify R² = 0.839 ± 0.001
    
    TODO: Calculate R² and check it's within tolerance
    
    Hints:
    - Use stats.linregress(pro_values, vol_values)
    - Calculate r_squared = r_value ** 2
    - Check abs(r_squared - 0.839) < 0.001
    """
    # TODO: Implement
    pass

def test_slope():
    """
    Verify slope = 0.712 ± 0.001
    
    TODO: Calculate slope and check it's within tolerance
    """
    # TODO: Implement
    pass

def test_organizations():
    """
    Verify correct organizations present
    
    TODO: Check volunteer orgs = {'OKCONCOM_WQX', 'CONSERVATION_COMMISSION'}
    TODO: Check professional orgs = {'OKWRB-STREAMS_WQX', 'O_MTRIBE_WQX'}
    """
    # TODO: Implement
    pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Learning Resources:**
- Pytest basics: https://docs.pytest.org/en/stable/getting-started.html
- Assert statements: https://docs.pytest.org/en/stable/assert.html

**Verification:**
- [ ] All tests implemented
- [ ] Run `pytest tests/test_pipeline.py -v`
- [ ] All tests pass ✅

---

## 📖 Task 7: Documentation

### Create `README.md`

```markdown
# Blue Thumb Virtual Triangulation Validation

[YOUR DESCRIPTION - Explain what this project does in your own words]

## Results

- N = [YOUR RESULT]
- R² = [YOUR RESULT]
- [YOUR INTERPRETATION]

## Quick Start

```bash
# Setup
git clone https://github.com/YOUR_USERNAME/bluethumb-validation.git
cd bluethumb-validation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python src/extract.py
python src/transform.py
python src/analysis.py
python src/visualize.py

# Verify
pytest tests/test_pipeline.py -v
```

## Methodology

[EXPLAIN YOUR APPROACH - What is virtual triangulation? How does it work?]

## Data Sources

[LIST YOUR DATA SOURCES]

## What I Learned

[WRITE ABOUT WHAT YOU LEARNED - This section is crucial for interviews!]

```

### Create `docs/REPLICATION.md`

**TODO:** Write step-by-step instructions for someone to reproduce your work.

Include:
- Prerequisites
- Installation steps
- Expected runtime for each step
- How to verify results
- Troubleshooting tips

---

## 🎯 Final Deliverables Checklist

Before considering this project complete:

### **Code Quality:**
- [ ] All functions have docstrings
- [ ] Code is readable (clear variable names)
- [ ] No hardcoded values (use config.yaml)
- [ ] Added comments for complex logic

### **Functionality:**
- [ ] extract.py downloads data successfully
- [ ] transform.py produces correct record counts
- [ ] analysis.py finds exactly 48 matches
- [ ] visualize.py creates clear plot
- [ ] All tests pass

### **Results (EXACT):**
- [ ] N = 48
- [ ] R² = 0.839 (±0.001)
- [ ] Slope = 0.712 (±0.001)
- [ ] Max distance ≤ 100m
- [ ] Max time ≤ 48 hours
- [ ] Min professional value > 25 mg/L

### **Column Names (CRITICAL):**
Your matched_pairs.csv MUST have these EXACT names:
```
Vol_SiteID, Pro_SiteID,
Vol_Organization, Pro_Organization,
Vol_Value, Pro_Value,
Vol_DateTime, Pro_DateTime,
Vol_Lat, Vol_Lon, Pro_Lat, Pro_Lon,
Distance_m, Time_Diff_hours
```

### **Documentation:**
- [ ] README explains what you built
- [ ] README shows your results
- [ ] REPLICATION guide has clear steps
- [ ] Code comments explain WHY not just WHAT

### **Git:**
- [ ] Repository is public on GitHub
- [ ] .gitignore excludes data files
- [ ] Commit messages are descriptive
- [ ] Repository structure is clean

---

## 🚀 Submission

When complete:

1. **Verify everything works:**
   ```bash
   # Fresh clone test
   cd /tmp
   git clone YOUR_REPO_URL
   cd bluethumb-validation
   # Follow your own README instructions
   # Do they work?
   ```

2. **Run final checks:**
   ```bash
   pytest tests/test_pipeline.py -v  # All pass?
   ls data/outputs/  # Files present?
   ```

3. **Submit:**
   - Repository URL
   - Screenshot of test results (all passing)
   - Brief description of what you learned

---

## 💡 What You're Learning

By building this yourself, you're learning:

✅ **API Integration** - Calling external APIs, handling responses  
✅ **Data Cleaning** - Filtering, null handling, outlier removal  
✅ **Spatial Algorithms** - Haversine distance calculation  
✅ **Algorithm Design** - Nested loops, optimization tradeoffs  
✅ **Statistical Analysis** - Linear regression, correlation  
✅ **Scientific Visualization** - Publication-quality plots  
✅ **Configuration Management** - YAML, parameterization  
✅ **Testing** - Unit tests, verification  
✅ **Documentation** - Clear explanations for others  
✅ **Git Workflow** - Version control, .gitignore  

**Most importantly:** You're learning how to build something from scratch, debug when it breaks, and verify it works.

---

## 🆘 When to Ask for Help

**Ask if:**
- You've been stuck on one error for >30 minutes
- Your results are way off (N≠48, R²<0.8, etc.)
- You don't understand what a function should do
- Tests fail and you can't figure out why

**Before asking:**
1. Read the error message carefully
2. Print intermediate values to debug
3. Check the hints and learning resources
4. Google the specific error

---

## ✨ Interview Talking Points

When you finish this project, you can say:

> "I built an ETL pipeline from scratch that validates citizen science data by matching volunteer measurements with professional monitoring using spatial-temporal algorithms. I implemented the Haversine formula to calculate distances between measurement locations, designed a matching algorithm that processes 330 million comparisons, and achieved a statistical correlation of R²=0.839 with 48 matched pairs. The project demonstrates my ability to work with geospatial data, implement mathematical algorithms, and build production-quality data pipelines."

**That's a portfolio piece you actually built and can explain in depth.**

---

## 📝 Final Notes

**This project is designed to be challenging but achievable.**

- You will get stuck. That's normal and good.
- You will make mistakes. That's how you learn.
- You will debug for hours. That's real development.
- You will feel accomplished when tests pass. Because you earned it.

**Take your time. Build it right. Understand each piece.**

When you're done, you'll have something real on your GitHub that you can talk about confidently in interviews.

Good luck! 🚀

---

**Last updated:** December 26, 2024  
**For:** Self-directed learning and portfolio development

