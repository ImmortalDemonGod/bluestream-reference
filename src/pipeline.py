"""
pipeline.py - Single entry point for BlueStream validation pipeline

Orchestrates the full Extract → Transform → Match → Statistics → Visualize workflow.
Reads all parameters from config/config.yaml. Produces official outputs in data/outputs/.

Usage:
    python src/pipeline.py                    # Run full pipeline
    python src/pipeline.py --skip-extract     # Skip EPA download (use cached raw data)

Expected output (Phase 2 - Volunteer Validation):
    data/outputs/matched_pairs.csv       (N=25)
    data/outputs/summary_statistics.txt  (R²=0.607)
    data/outputs/validation_plot.png
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def load_config():
    """Load configuration from YAML file"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def get_git_hash():
    """Get current git commit hash, or 'unknown' if not in a git repo"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def file_sha256(filepath):
    """Compute SHA-256 hash of a file"""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def run_extract(config):
    """Step 1: Download raw data from EPA WQP + ArcGIS volunteer data"""
    from src.extract import download_oklahoma_chloride, download_arcgis_volunteer_data
    print("\n" + "=" * 60)
    print("STEP 1: DATA EXTRACTION")
    print("=" * 60)
    download_oklahoma_chloride(config)
    download_arcgis_volunteer_data(config)


def run_transform(config):
    """Step 2: Clean data, separate volunteer/professional, apply Blue Thumb override"""
    from src.transform import process_data, save_processed_data

    print("\n" + "=" * 60)
    print("STEP 2: DATA TRANSFORMATION")
    print("=" * 60)

    volunteer_df, professional_df = process_data(config)
    save_processed_data(volunteer_df, professional_df, config)

    return volunteer_df, professional_df


def run_analysis(config):
    """Step 3: Spatial-temporal matching — both pro-to-pro and vol-to-pro"""
    from src.analysis import run_comparison, save_results, run_spatial_coverage

    print("\n" + "=" * 60)
    print("STEP 3: SPATIAL-TEMPORAL MATCHING")
    print("=" * 60)

    proc_dir = Path(config['output_paths']['processed_data'])

    # Load professional reference (same for both comparisons)
    professional_df = pd.read_csv(proc_dir / "professional_chloride.csv")
    professional_df['ActivityStartDate'] = pd.to_datetime(professional_df['ActivityStartDate'])

    # --- Pro-to-Pro baseline ---
    p2p_stats = None
    rb_path = proc_dir / "rotating_basin_chloride.csv"
    if rb_path.exists():
        rb_df = pd.read_csv(rb_path)
        rb_df['ActivityStartDate'] = pd.to_datetime(rb_df['ActivityStartDate'])
        _, p2p_stats = run_comparison(
            "PRO-TO-PRO: OCC Rotating Basin vs Professional Reference",
            rb_df, professional_df, config, "pro_to_pro"
        )

    # --- Vol-to-Pro validation ---
    volunteer_df = pd.read_csv(proc_dir / "volunteer_chloride.csv")
    volunteer_df['ActivityStartDate'] = pd.to_datetime(volunteer_df['ActivityStartDate'])
    v2p_matches, v2p_stats = run_comparison(
        "VOL-TO-PRO: Blue Thumb Volunteers vs Professional Reference",
        volunteer_df, professional_df, config, "vol_to_pro"
    )

    # Save vol-to-pro as default matched_pairs.csv for backward compatibility
    if len(v2p_matches) > 0:
        save_results(v2p_matches, v2p_stats, config)

    # Side-by-side summary
    print(f"\n{'=' * 60}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Metric':<25} {'Pro-to-Pro':>15} {'Vol-to-Pro':>15}")
    print(f"{'-' * 55}")
    if p2p_stats and v2p_stats:
        print(f"{'N':<25} {p2p_stats['n']:>15} {v2p_stats['n']:>15}")
        print(f"{'R²':<25} {p2p_stats['r_squared']:>15.3f} {v2p_stats['r_squared']:>15.3f}")
        print(f"{'Slope':<25} {p2p_stats['slope']:>15.3f} {v2p_stats['slope']:>15.3f}")

    # Spatial coverage analysis
    run_spatial_coverage(volunteer_df, professional_df, v2p_matches, config)

    return v2p_matches, v2p_stats


def run_visualize(config):
    """Step 4: Generate validation plot"""
    from src.visualize import create_validation_plot

    print("\n" + "=" * 60)
    print("STEP 4: VISUALIZATION")
    print("=" * 60)

    results_dir = Path(config['output_paths']['results'])
    matches_df = pd.read_csv(results_dir / "matched_pairs.csv")
    create_validation_plot(matches_df, config)


def write_metadata(config, stats_dict):
    """Write reproducibility manifest alongside official outputs"""
    output_dir = Path(config['output_paths']['results'])
    matched_csv = output_dir / "matched_pairs.csv"

    metadata = {
        'generated_at': datetime.now().isoformat(),
        'git_commit': get_git_hash(),
        'config_hash': file_sha256('config/config.yaml'),
        'matched_pairs_hash': file_sha256(str(matched_csv)) if matched_csv.exists() else None,
        'python_version': sys.version,
        'matching_parameters': config['matching_parameters'],
        'results': {
            'n': stats_dict['n'],
            'n_unique_sites': stats_dict.get('n_unique_sites'),
            'r_squared': round(stats_dict['r_squared'], 4),
            'slope': round(stats_dict['slope'], 4),
            'p_value': float(f"{stats_dict['p_value']:.6e}"),
        }
    }

    ext_path = config.get('external_sources', {}).get('volunteer_blue_thumb_csv')
    if ext_path and Path(ext_path).exists():
        metadata['volunteer_csv_hash'] = file_sha256(ext_path)

    meta_path = output_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata manifest: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description='BlueStream Validation Pipeline')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip EPA download, use cached raw data')
    args = parser.parse_args()

    config = load_config()

    print("=" * 60)
    print("BLUESTREAM VALIDATION PIPELINE")
    print(f"  Mode: Volunteer Validation (Phase 2)")
    print(f"  Parameters: {config['matching_parameters']['max_distance_meters']}m / "
          f"{config['matching_parameters']['max_time_hours']}h / "
          f"{config['matching_parameters']['match_strategy']}")
    print(f"  Git: {get_git_hash()}")
    print("=" * 60)

    if not args.skip_extract:
        run_extract(config)
    else:
        print("\n  Skipping extraction (--skip-extract)")

    run_transform(config)
    matches_df, stats_dict = run_analysis(config)
    run_visualize(config)
    write_metadata(config, stats_dict)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    n_sites = stats_dict.get('n_unique_sites', '?')
    print(f"  N = {stats_dict['n']} matches from {n_sites} sites, "
          f"R² = {stats_dict['r_squared']:.3f}, Slope = {stats_dict['slope']:.3f}")
    print(f"  Outputs: {config['output_paths']['results']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
