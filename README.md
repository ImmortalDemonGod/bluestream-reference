# BlueStream: Forensic Validation of Citizen Science Data
 
 > **Project:** Oklahoma Blue Thumb Data Validation Pipeline
 > **Engineering Lead:** [Your Name]
 > **Architected by:** Miguel Ingram (Black Box Research)
 
 ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Validation_Complete-success)
 
 ## Executive Summary
 
 This repository implements an end-to-end ETL pipeline to perform a forensic audit of the **Oklahoma Blue Thumb** volunteer water quality monitoring program.
 
 By mining 30 years of historical records from the **EPA Water Quality Portal (WQP)**, we apply **Virtual Triangulation**: matching volunteer measurements with professional agency measurements (USGS, OWRB, Tribal) taken at the same location within a 48-hour window.
 
 **Primary finding:** Blue Thumb volunteer data correlates with professional measurements at **`R² = 0.839`**, validating the program’s rigor and showing that properly trained citizen scientists can track environmental signals with professional-grade fidelity.
 
 ## Key Results
 
 The pipeline processes ~50,000 records to identify strict spatial-temporal matches.
 
 | Metric | Result | Interpretation |
 | :--- | :--- | :--- |
 | **Correlation (`R²`)** | **0.839** | Strong agreement between volunteer and professional measurements. |
 | **Sample Size (`N`)** | **48** | Matched pairs within 100m / 48hrs. |
 | **Slope** | **0.712** | Volunteers measure systematically lower (~29%), consistent with calibration/methodology differences (bias), not random error (noise). |
 | **P-Value** | **< 0.0001** | Relationship is statistically significant. |
 
 ### Validation Visualization
 
 ![Validation Plot](data/outputs/validation_plot.png)
 
 - **Note:** `data/outputs/` is gitignored by default. This repo includes a narrow exception in `.gitignore` to allow committing `data/outputs/validation_plot.png` (while continuing to ignore other outputs).
 
 ## Technical Architecture
 
 This repository implements a reproducible **Functional ETL** architecture designed for auditability and easy re-runs.
 
 All pipeline parameters are centralized in `config/config.yaml`.
 
 ### 1. Extraction (`src/extract.py`)
 
 - Interfaces with the **EPA WQP API**.
 - Uses `dataProfile='resultPhysChem'` to keep schemas consistent.
 - Handles large exports via ZIP download + extraction.
 
 ### 2. Transformation (`src/transform.py`)
 
 - Normalizes schemas across providers (STORET vs. NWIS).
 - Filters to the target characteristic (Chloride) and performs quality controls (invalid coordinates, non-detect handling).
 - Splits datasets into volunteer vs. professional organizations.
 
 ### 3. Analysis (`src/analysis.py`)
 
 - Implements Virtual Triangulation using **KD-Tree spatial indexing** (`scipy.spatial.cKDTree`).
 - Reduces matching complexity from `O(N × M)` (brute force) to approximately `O(N log M)`.
 - Applies a rigorous filter:
   - **Distance:** Haversine distance ≤ 100 meters
   - **Time:** Δt ≤ 48 hours
   - **Match strategy:** Configurable via `matching_parameters.match_strategy`
 
 ## Quick Start
 
 ### Prerequisites
 
 - Python 3.9+
 - Virtual environment recommended
 
 ### Installation
 
 ```bash
 # Clone the repository
 git clone https://github.com/[YOUR_USERNAME]/bluestream-test.git
 cd bluestream-test
 
 # Install dependencies
 python -m venv venv
 source venv/bin/activate  # Windows: venv\Scripts\activate
 pip install -r requirements.txt
 ```
 
 ### Running the Pipeline
 
 ```bash
 # 1. Download raw data from EPA (5-10 mins)
 python src/extract.py
 
 # 2. Clean and structure data
 python src/transform.py
 
 # 3. Run spatial-temporal matching algorithm
 python src/analysis.py
 
 # 4. Generate visualization
 python src/visualize.py
 ```
 
 ### Verification
 
 ```bash
 pytest tests/test_pipeline.py -v
 ```
 
 ## Why This Matters
 
 Citizen science is often dismissed as “hobbyist data.” This project shows that with the right data infrastructure, volunteer monitoring can be rigorously validated.
 
 Oklahoma’s data infrastructure is unusually well-suited for this audit because volunteer organizations are represented with stable organization identifiers in EPA’s WQX/WQP metadata. That makes Virtual Triangulation possible at scale.
 
 ## Limitations and Responsible Use
 
 - **Unit normalization:** The current pipeline assumes units are consistent across matched measurements.
 - **Screening-level interpretation:** This analysis is designed for validation and QA/QC screening, not regulatory determination. It should not be used to make public accusations about impairment status or compliance.
 
 ## License
 
 Distributed under the MIT License. See `LICENSE` for more information.
 
 **Disclaimer:** This is an independent analysis conducted by Black Box Research volunteers. It is not an official publication of the Oklahoma Conservation Commission.
