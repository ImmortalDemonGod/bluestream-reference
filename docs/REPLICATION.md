# Replication Guide

**TODO:** Write step-by-step instructions for someone to reproduce your work.

## Prerequisites
- Python 3.x
- Git

## Installation
1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`

## Running the Pipeline

### Step 1: Extraction
- Command: `python src/extract.py`
- Expected runtime: ~5-10 minutes
- Output: `data/raw/oklahoma_chloride.csv`

### Step 2: Transformation
- Command: `python src/transform.py`
- Expected runtime: ~1 minute
- Outputs: 
  - `data/processed/volunteer_chloride.csv`
  - `data/processed/professional_chloride.csv`

### Step 3: Analysis (Virtual Triangulation)
- Command: `python src/analysis.py`
- Expected runtime: ~30-60 minutes
- Output: `data/outputs/matched_pairs.csv`

### Step 4: Visualization
- Command: `python src/visualize.py`
- Output: `data/outputs/validation_plot.png`

## Verification
Run the test suite to verify results:
```bash
pytest tests/test_pipeline.py -v
```

## Troubleshooting
- If download fails...
- If matches are 0...
