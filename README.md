 # BlueStream: Forensic Validation of Citizen Science Data 
 
 > **Principal Investigator:** Miguel Ingram (Black Box Research Labs)
 > **Status:** Collaborative Validation Study | Phase 1 (Aggregate Analysis) Complete
 > **Institutional Context:** Directed research in alignment with the **Oklahoma Conservation Commission (OCC)**
 > **Milestone:** Methodology and findings reviewed with OCC Leadership (Jan 2026)
 > **Validation Target:** N=48, R²=0.839 (Verified)
 
 ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Regulatory_Reference-success)
 
 ## 📊 Executive Summary
 
 This repository contains the **Reference Implementation** for the BlueStream validation protocol. It serves as the forensic audit engine for the **Oklahoma Blue Thumb** volunteer water quality monitoring program.
 
 Designed and architected by **Miguel Ingram**, this pipeline mines 30 years of historical data from the EPA Water Quality Portal to perform **"Virtual Triangulation"**—a rigorous spatial-temporal matching algorithm that validates volunteer measurements against professional agency sensors.
 
 **The Finding:** Blue Thumb volunteer data correlates with professional sensors at **$R^2 = 0.839$**, proving that the program's data infrastructure produces professional-grade environmental intelligence.
 
 ---
 
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
 git clone https://github.com/ImmortalDemonGod/bluestream-test.git
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

---

### 🏛️ Institutional Context & Data Governance

This research is conducted by **Black Box Research Labs LLC** with technical support and data access provided by the **Oklahoma Conservation Commission (OCC)**.

**Current Trajectory:**
- **Endorsement:** Letter of Support in development (OCC)
- **Validation:** Findings scheduled for the **2026 OCLWA Conference**
- **Regulatory Alignment:** Analysis methodology reviewed with OCC Blue Thumb program leadership (Jan 15, 2026)

This repository serves as the forensic engine for the study. All data handling and algorithmic verification are performed using the **AIV (Algorithmic Intelligence Validation) Protocol** to ensure scientific defensibility and program-specific alignment.

**Black Box Research Labs** specializes in forensic data validation and algorithmic audit systems for environmental monitoring programs.
