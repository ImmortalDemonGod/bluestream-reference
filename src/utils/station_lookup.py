"""
station_lookup.py - WQP Station API utilities

Extracted from src/analysis.py to separate HTTP/API concerns from the
core matching algorithm. These functions query the EPA Water Quality Portal
Station endpoint to resolve site IDs to human-readable station names.

Usage:
    from src.utils.station_lookup import fetch_wqp_station_profiles
    from src.utils.station_lookup import station_name_lookup_from_matched_pairs
"""

import pandas as pd
import requests


def _chunk_list(items, chunk_size):
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def fetch_wqp_station_profiles(site_ids, timeout_seconds=30, chunk_size=50):
    """Fetch station metadata from WQP Station API for a list of site IDs.

    Args:
        site_ids: List of MonitoringLocationIdentifier strings
        timeout_seconds: HTTP request timeout
        chunk_size: Number of site IDs per API call

    Returns:
        DataFrame with station metadata (name, org, lat/lon, HUC, county, state)
    """
    site_ids = [s for s in site_ids if isinstance(s, str) and s.strip()]
    site_ids = sorted(set(site_ids))
    if not site_ids:
        return pd.DataFrame()

    base_url = "https://www.waterqualitydata.us/data/Station/search"
    dfs = []

    for chunk in _chunk_list(site_ids, chunk_size=chunk_size):
        params = [("siteid", sid) for sid in chunk]
        params.append(("mimeType", "geojson"))

        response = requests.get(base_url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()

        features = data.get("features") if isinstance(data, dict) else None
        if not isinstance(features, list):
            raise ValueError(
                f"Unexpected WQP Station response. Top-level keys: "
                f"{sorted(list(data.keys())) if isinstance(data, dict) else type(data)}"
            )

        records = []
        for feature in features:
            props = feature.get("properties") if isinstance(feature, dict) else None
            if not isinstance(props, dict):
                continue

            lat = props.get("LatitudeMeasure")
            lon = props.get("LongitudeMeasure")
            geometry = feature.get("geometry") if isinstance(feature, dict) else None
            if (lat is None or lon is None) and isinstance(geometry, dict):
                coords = geometry.get("coordinates")
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    lon = lon if lon is not None else coords[0]
                    lat = lat if lat is not None else coords[1]

            records.append(
                {
                    "MonitoringLocationIdentifier": props.get("MonitoringLocationIdentifier"),
                    "MonitoringLocationName": props.get("MonitoringLocationName"),
                    "OrganizationIdentifier": props.get("OrganizationIdentifier"),
                    "OrganizationFormalName": props.get("OrganizationFormalName"),
                    "ProviderName": props.get("ProviderName"),
                    "LatitudeMeasure": lat,
                    "LongitudeMeasure": lon,
                    "HUC": props.get("HUC"),
                    "CountyName": props.get("CountyName"),
                    "StateCode": props.get("StateCode"),
                }
            )

        dfs.append(pd.DataFrame(records))

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not df.empty:
        df = df.drop_duplicates(subset=["MonitoringLocationIdentifier"], keep="first")
    return df


def station_name_lookup_from_matched_pairs(matched_pairs_csv_path, timeout_seconds=30):
    """Look up station names for all sites in a matched_pairs.csv file.

    Args:
        matched_pairs_csv_path: Path to matched_pairs.csv
        timeout_seconds: HTTP request timeout

    Returns:
        DataFrame with station metadata for all volunteer and professional sites
    """
    pairs = pd.read_csv(matched_pairs_csv_path, low_memory=False)
    missing = [c for c in ["Vol_SiteID", "Pro_SiteID"] if c not in pairs.columns]
    if missing:
        raise ValueError(f"matched_pairs.csv missing columns: {missing}")

    def _full_siteid(org_id, site_id):
        if not isinstance(site_id, str) or not site_id.strip():
            return None
        if not isinstance(org_id, str) or not org_id.strip():
            return site_id.strip()
        org_id = org_id.strip()
        site_id = site_id.strip()
        if site_id.startswith(org_id + "-"):
            return site_id
        return org_id + "-" + site_id

    vol_siteids = pairs.apply(
        lambda r: _full_siteid(r.get("Vol_Organization"), r.get("Vol_SiteID")), axis=1
    )
    pro_siteids = pairs.apply(
        lambda r: _full_siteid(r.get("Pro_Organization"), r.get("Pro_SiteID")), axis=1
    )
    site_ids = (
        pd.concat([vol_siteids, pro_siteids], ignore_index=True)
        .dropna()
        .astype(str)
        .tolist()
    )
    return fetch_wqp_station_profiles(site_ids=site_ids, timeout_seconds=timeout_seconds)
