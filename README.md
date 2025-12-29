# Blue Thumb Virtual Triangulation Validation

This project implements an end-to-end ETL pipeline that downloads Oklahoma water quality results from the EPA Water Quality Portal, filters to chloride measurements, and performs **virtual triangulation**: spatial-temporal matching of volunteer samples to nearby professional samples to validate measurement fidelity.

## Results

- N = 48 matched pairs
- R² = 0.839
- Slope = 0.712

## Quick Start

```bash
# Setup
git clone <your-repo-url>
cd <your-repo-folder>
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

- **Extract**: Download `resultPhysChem` data for Oklahoma from the Water Quality Portal.
- **Transform**: Filter to chloride, clean coordinates/values, split volunteer vs professional organizations.
- **Analyze**: Match volunteer samples to professional samples within `100m` and `48h`.

Key knobs live in `config/config.yaml`:

- `matching_parameters.match_strategy`
  - `all`: include all qualifying matches (reproduces the 48-pair target)
  - `closest`: choose the closest professional match per volunteer sample
- `matching_parameters.min_concentration_mg_l`: applied to professional results only

## Data Sources

- EPA Water Quality Portal (WQP)

## What I Learned

- Building an ETL pipeline with reproducible configuration.
- Implementing spatial-temporal matching and optimizing it with spatial indexing.
- Validating results with automated tests.
