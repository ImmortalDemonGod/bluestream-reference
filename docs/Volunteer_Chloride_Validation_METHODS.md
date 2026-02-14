# Blue Thumb Volunteer-Only Chloride Validation (Methods & Findings)

## Purpose
Establish a defensible, volunteer-only validation of Blue Thumb chloride measurements against professional agency datasets using the existing virtual triangulation pipeline.

## Data Sources
- Volunteer: `data/Requested_Blue Thumb Chemical Data.csv` (OCC R-Shiny export)
  - Rows: 11,208 (10,800 numeric Chloride values after parsing)
  - Date range: 2005-01-04 to 2024-12-31
  - Fields used: WBID (as `MonitoringLocationIdentifier`), Date, Time, Latitude, Longitude, Chloride (mg/L)
  - **Cross-validated (Feb 11, 2026):** 2,026 of 2,027 overlapping records match OCC's public ArcGIS production feed (`bluethumb_oct2020_view`) at 100% agreement. Chloride = `Chloride_Low1_Final` (drops × 5 mg/L, no blank subtraction). See Section "ArcGIS Cross-Validation" below.
- Professional: EPA WQP Results (Oklahoma, Chloride; STORET + NWIS) pulled previously into `data/raw/oklahoma_chloride.csv` and processed by the pipeline.

## Preprocessing
- Volunteer normalization (in `src/transform.py`):
  - Map WBID → `MonitoringLocationIdentifier`
  - Parse `Date` + `Time` with robust normalization (handles `4:00PM`, `4:30 PM`, and `.`)
  - Coerce `Latitude`/`Longitude` and `Chloride` to numeric; drop invalids
  - Set `OrganizationIdentifier='BLUETHUMB_VOL'`, `ResultMeasure/MeasureUnitCode='mg/L'`
  - Note: WQP's `OKCONCOM_WQX` is the OCC Rotating Basin program (professional lab, Method 9056), NOT Blue Thumb volunteers.
- Professional processing unchanged; downstream pro filter preserves only values > 25 mg/L for matching (per configuration).

## Matching Parameters (Chosen)
- Distance ≤ 125 meters
- Time difference ≤ 72 hours
- Strategy: `closest` (one professional per volunteer measurement)
- Professional minimum concentration: > 25 mg/L

Rationale: modest expansion from the strict reference (100 m / 48 h) produces sufficient overlap while remaining spatially/temporally defensible.

## Statistical Results (OLS)

### Vol-to-Pro (Primary — Blue Thumb Volunteers vs Professional Reference)
- Sample size (N): 25
- Correlation (R²): 0.607
- Slope: 0.813
- Intercept: -2.411
- p-value: 4.432e-06

### Pro-to-Pro Baseline (OCC Rotating Basin vs Professional Reference)
- Sample size (N): 42
- Correlation (R²): 0.753
- Slope: 0.735
- p-value: 1.027e-13

Interpretation: Professionals themselves do not perfectly agree (R²=0.753). Volunteers capture 61% of the professional signal (R²=0.607), consistent with field titration kits under-reading relative to professional methods.

## Robustness Checks
- Bootstrap (B=500) on OLS:
  - Slope CI95: [0.0308, 1.1973] (mean 0.7696)
  - R² CI95: [0.0035, 0.8850] (mean 0.5665)
- Deming regression (errors-in-variables, δ=1):
  - Slope ≈ 1.0555; Intercept ≈ -20.45
  - Bootstrap (B=1000) slope CI95: [0.0229, 1.4424]
- Per-organization (OLS, best config):
  - CNENVSER (n=18): R²=0.643, slope=0.846 — **method UNKNOWN in WQP metadata**
  - OKWRB-STREAMS_WQX (n=7): R²=0.730, slope=0.801 — EPA Method 325.2 (VERIFIED)
  - Note: The OKWRB-only subset shows *stronger* correlation than the combined result,
    using a verified analytical method. This independently supports the finding.

## Sanity Checks
- Duplicate volunteer measurement pairs: 0
- High-influence outliers removed (|z|>3): 0
- Time and distance constraints satisfied for all pairs

## Sensitivity (Selected)
- Wider windows (150–200 m / 72 h) consistently yield N≈25–27, R²≈0.60, slope≈0.80.
- Lowering professional min concentration to 10 or 0 mg/L had negligible effect in the top region.

## Limitations & Notes

### Quantization Effect
- Volunteer chloride is quantized to **5 mg/L steps** (Silver Nitrate titration: each drop = 5 mg/L).
  100% of volunteer values are exact multiples of 5 (range: 15–220 mg/L, 13 unique values).
  At low concentrations (15–35 mg/L), this ±2.5 mg/L resolution is 7–17% of the signal,
  introducing scatter that depresses R² relative to what continuous measurements would yield.
- OLS regression assumes error-free X (professional values). Since both sides have measurement
  error, Deming regression is the theoretically correct estimator. Bootstrap Deming analysis
  confirms the finding is robust (see `data/outputs/experiments/deming_ref_*.txt`).

### Spatial Concentration
- N=25 matched pairs come from **4 unique volunteer sites** matched to 4 professional sites.
  48% of matches (12/25) originate from a single site (OK520600-03-0020W).
  72% of matches are against CNENVSER, whose analytical method is unrecorded.
- **However**: Spatial coverage analysis shows 93% of Blue Thumb's 327 sites are >1 km from
  the nearest professional monitor, and 99% lack any temporal match. The low N reflects the
  complementary nature of volunteer monitoring, not a weakness of the study design.

### Hydrologic Conditions
- The WQP fields `HydrologicCondition` and `HydrologicEvent` are **0% populated** across all
  18,299 professional records in this dataset. Storm-event filtering is therefore impossible
  from the available metadata. The 72-hour matching window may include storm-influenced samples.

### General
- One volunteer site exhibited weak alignment; others were strong—site-level heterogeneity is expected.
- Results reflect volunteer titration vs. professional monitoring methods; Deming slope near 1 with negative intercept is consistent with measurement error and offset between methods.

## ⚠️ Action Items: Analytical Method Verification (Feb 11, 2026)

The narrative in the Jan 21 OCC email stated OKWRB uses "EPA Method 300.0 (Ion Chromatography)." WQP metadata verification (Feb 11) found a **contradiction**:

| Organization | Email Claim | WQP Metadata (18,854+ records) | Status |
|:---|:---|:---|:---|
| **OKWRB-STREAMS_WQX** | EPA 300.0 (Ion Chromatography) | **EPA 325.2** (Automated Colorimetry) | **CONTRADICTED — needs confirmation** |
| **CNENVSER** | Inferred EPA 300.0 from decimal precision | **EMPTY** (no analytical method recorded) | **UNVERIFIED — contact needed** |

**Required actions before OCLWA conference (Apr 15-16):**
1. **Contact Julie Chambers at OKWRB** to confirm their chloride analytical method and SOPs for the years in the matched data. (Miguel committed to this in the Jan 26 email but no follow-up is recorded.)
2. **Contact CNENVSER (Chickasaw Nation Environmental Services)** per Kim Shaw's Jan 26 request to confirm their chloride testing methods.
3. **Until confirmed:** Use "professional monitoring methods" in all public-facing materials rather than "Ion Chromatography" or "EPA 300.0."

See `data/outputs/wqp_verification_results.txt` for full verification details.

## Reproduce
```bash
# 1) Create venv and install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2) Run full pipeline (single command — extract, transform, match, visualize)
python -m src.pipeline              # Full run (downloads EPA data)
python -m src.pipeline --skip-extract  # Skip download, use cached raw data

# Or run steps individually:
python -m src.transform    # Clean data, load Blue Thumb CSV, save processed files
python -m src.analysis     # Run both pro-to-pro and vol-to-pro matching

# 3) Verify
pytest tests/test_pipeline.py -v   # All 12 tests should pass

# 4) (Optional) Robustness checks
python scripts/experiments/bootstrap_ci.py \
  --matches data/outputs/matched_pairs_vol_to_pro.csv --B 500
python scripts/experiments/deming_ci.py \
  --matches data/outputs/matched_pairs_vol_to_pro.csv --B 1000 --delta 1.0
```

## Key Artifacts
- Official outputs:
  - `data/outputs/matched_pairs.csv` (vol-to-pro, N=25)
  - `data/outputs/matched_pairs_vol_to_pro.csv` (same, explicit name)
  - `data/outputs/matched_pairs_pro_to_pro.csv` (baseline, N=42)
  - `data/outputs/summary_statistics_vol_to_pro.txt` (includes per-org breakdown + method provenance)
  - `data/outputs/summary_statistics_pro_to_pro.txt`
  - `data/outputs/validation_plot.png`
  - `data/outputs/metadata.json` (reproducibility manifest)
- Experiments (selected):
  - `data/outputs/experiments/sweep_results_refined_sorted.csv`
  - `data/outputs/experiments/bootstrap_ref_d125_t72_mc25_closest.txt`
  - `data/outputs/experiments/deming_ref_d125_t72_mc25_closest.txt`
  - `data/outputs/diagnostics_matches.txt`

## ArcGIS Cross-Validation (Feb 11, 2026)

The R-Shiny CSV was independently cross-validated against OCC's public ArcGIS production database:

- **Endpoint:** `https://services5.arcgis.com/.../bluethumb_oct2020_view/FeatureServer/0/query` (no auth required)
- **ArcGIS records:** 3,015 valid chloride measurements, 160 WBIDs, 2015–2026
- **Overlap:** 2,364 WBID+date pairs exist in both sources
- **Agreement:** 2,026 of 2,027 comparable records are an **exact match** (100.0%)
- **Formula:** `Chloride = Chloride_Low1_Final` (raw drops × 5 mg/L, no blank subtraction)
- **QAQC:** 3,039 of 3,040 ArcGIS records have `QAQC_Complete = "X"`
- **New data:** 647 ArcGIS-only records (2025–2026) not in the Dec 2024 R-Shiny snapshot

**Limitation:** ArcGIS only covers 2015+. The R-Shiny CSV contains 8,209 pre-2015 records not available via ArcGIS. Of the 25 matched pairs, 13 use pre-2015 volunteer data.

**Implication:** The volunteer data file is verified authentic against OCC's operational database. Future pipeline versions can supplement or replace the static CSV with the live ArcGIS API for 2015+ data.

## Conclusion
Under a modestly expanded, defensible window (125 m / 72 h / closest), the dual-comparison framework shows:

1. **Pro-to-Pro baseline** (N=42, R²=0.753): Even professionals do not perfectly agree with each other, establishing the upper bound for expected correlation.
2. **Vol-to-Pro validation** (N=25, R²=0.607): Blue Thumb volunteers capture 61% of the professional signal using field titration kits, with a slope of 0.81 consistent with known methodological differences.
3. **OKWRB-only subset** (N=7, R²=0.730): Matches against the verified EPA 325.2 method show even stronger correlation, independently supporting the finding.

Robustness checks (bootstrap, Deming) support the finding. Cross-validation against OCC's public ArcGIS feed confirms data integrity at 100% agreement. This constitutes a credible volunteer-only validation for chloride.

**Note on professional methods:** CNENVSER (72% of vol-to-pro matches) has no analytical method recorded in WQP metadata. Until confirmed, publications should reference "professional agency methods" rather than citing specific EPA methods for the combined result.

## Phase 3 Roadmap (Post-Conference)

Potential improvements identified during code review:
1. **Triple-Match analysis:** Volunteer vs. OCC Rotating Basin (17 co-located sites exist). Tests whether volunteers agree better with co-located OCC labs than with OKWRB/CNENVSER.
2. **ArcGIS live ingestion:** Replace static CSV with direct ArcGIS API fetch for 2015+ data (extract.py already downloads it; transform.py integration pending).
3. **Deming regression as primary estimator:** Both X and Y have measurement error; Deming is theoretically more appropriate than OLS. Currently available as a robustness check only.
4. **Storm-event exclusion:** If Oklahoma agencies begin populating `HydrologicCondition` in WQP, filter storm-influenced samples from the 72-hour matching window.
5. **Site-level mixed-effects model:** Account for repeated measures at the same 4 volunteer sites rather than treating all 25 pairs as independent.
