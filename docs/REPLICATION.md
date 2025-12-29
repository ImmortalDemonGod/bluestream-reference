# Replication Guide

This guide documents how to reproduce the pipeline outputs from scratch.

## Prerequisites
- Python 3.x
- Git

## Installation
1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`

## Running the Pipeline

Before you run, confirm the key settings in `config/config.yaml`:

- `matching_parameters.match_strategy: "all"`
- `matching_parameters.min_concentration_mg_l: 25`

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
- Expected runtime: typically seconds (uses spatial indexing)
- Output: `data/outputs/matched_pairs.csv`

### Step 4: Visualization
- Command: `python src/visualize.py`
- Output: `data/outputs/validation_plot.png`

## Verification
Run the test suite to verify results:
```bash
pytest tests/test_pipeline.py -v
```

Expected output metrics (with the config above):

- N = 48
- R² = 0.839
- Slope = 0.712

## Troubleshooting
- If the download fails, re-run `python src/extract.py` and confirm you have network access.
- If matches are unexpectedly low, confirm `match_strategy` and the concentration threshold in `config/config.yaml`.
