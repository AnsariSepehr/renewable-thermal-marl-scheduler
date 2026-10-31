"""
Phase 1.5 -- Real solar data integration (PVGIS)
==================================================
Replaces the purely-synthetic solar curves from Phase 1 with real, physically
grounded ones, pulled from PVGIS (the EU Commission's Photovoltaic
Geographical Information System): https://re.jrc.ec.europa.eu/

Why PVGIS specifically:
- Free, no API key or account needed.
- Works for any latitude/longitude on Earth (not just Europe).
- Built from satellite-derived irradiance + a "typical meteorological year",
  so it captures real seasonal/latitude/climate effects, not just a bell curve.

How we use it:
1. We assign each of our 10 datacenters a *real* city location, spread across
   different latitudes and climates -- this is itself an improvement over the
   original paper, which used one single real PV site (Gatton, Australia) for
   every datacenter regardless of location.
2. For each location we request one year of hourly PV output for a reference
   1 kWp installation (`peakpower=1`), which gives us a realistic *shape*
   (day/night, seasonal, weather-driven variability) independent of size.
3. We average that year into a typical 24-hour diurnal profile (per hour of
   day), normalize it to [0, 1], then scale it to a target peak wattage per
   datacenter -- keeping magnitudes comparable to our server power draws
   (same range used in Phase 1's synthetic version), while the *shape* is now
   real.
4. Results are cached to disk (`data/pvgis_cache/`) so we only hit the API
   once per location.

Fallback: if PVGIS is unreachable (offline, blocked network, API hiccup) we
fall back to the Phase-1 synthetic bell-curve generator for that location, and
print a clear warning -- so the notebook never silently breaks, it just loses
the "real data" benefit until network access is available.
"""

import os
import json
import numpy as np
import pandas as pd
import requests

from .datacenter_env import make_solar_profile  # Phase 1 synthetic fallback

PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
CACHE_DIR = os.path.join("data", "pvgis_cache")

# 10 real-world datacenter-style locations, spread across climates/latitudes
# on purpose -- this heterogeneity is exactly what makes green-aware
# scheduling interesting (some DCs are sunny/green at times others aren't).
DATACENTER_LOCATIONS = [
    ("Frankfurt_DE",    50.11,   8.68),
    ("N_Virginia_US",   38.95, -77.45),
    ("Singapore_SG",     1.35, 103.82),
    ("Sydney_AU",      -33.87, 151.21),
    ("Tokyo_JP",         35.68, 139.65),
    ("Sao_Paulo_BR",    -23.55, -46.63),
    ("Mumbai_IN",        19.08,  72.88),
    ("Cape_Town_ZA",    -33.92,  18.42),
    ("London_UK",        51.51,  -0.13),
    ("Oregon_US",        45.52, -122.68),
]


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.csv")


def fetch_pvgis_hourly(lat: float, lon: float, year: int = 2019) -> pd.DataFrame:
    """Pulls one year of hourly PV power output (Watts, for a reference 1 kWp
    system) from PVGIS for the given coordinates. Raises on failure -- caller
    decides how to handle it (we use a synthetic fallback, see below).

    PVGIS auto-selects a radiation database by region, and its default
    (SARAH2/3, Europe+Africa+Asia coverage) returns HTTP 400 for locations
    outside that footprint -- notably the Americas. So for longitudes in the
    Americas we explicitly request the NSRDB database instead, and only fall
    back to letting PVGIS auto-select if that also fails.
    """
    base_params = {
        "lat": lat, "lon": lon,
        "outputformat": "json",
        "pvcalculation": 1,
        "peakpower": 1,       # 1 kWp reference system -> output is a normalized "shape"
        "loss": 14,            # typical system losses (%), PVGIS default recommendation
        # NOTE: startyear/endyear are set per-database below (db_year_override),
        # since each PVGIS database only covers a fixed historical window and
        # the `year` argument is not used directly for this reason.
    }

    # Try the region-appropriate database first, then fall back to no explicit
    # database (let PVGIS auto-select), trying whichever we haven't tried yet.
    # Each database only has data for a fixed historical window, so the
    # requested year has to match the database or PVGIS returns 400:
    #   PVGIS-NSRDB  : 2005-2015 only (Americas)
    #   PVGIS-SARAH2 : 2005-2020      (Europe/Africa/most of Asia)
    is_americas = lon < -30
    db_attempts = ["PVGIS-NSRDB", None] if is_americas else [None, "PVGIS-SARAH2"]
    db_year_override = {"PVGIS-NSRDB": 2015, "PVGIS-SARAH2": 2019, None: 2019}

    last_error = None
    for db in db_attempts:
        params = dict(base_params)
        params["startyear"] = params["endyear"] = db_year_override[db]
        if db is not None:
            params["raddatabase"] = db
        try:
            resp = requests.get(PVGIS_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            hourly = data["outputs"]["hourly"]
            df = pd.DataFrame(hourly)
            # PVGIS timestamps look like '20190101:0010' -> extract hour of day
            df["hour"] = df["time"].str.slice(9, 11).astype(int)
            return df[["time", "hour", "P"]]  # P = PV power output in Watts (for 1 kWp)
        except Exception as e:
            last_error = e
            continue
    raise last_error


def typical_daily_shape(hourly_df: pd.DataFrame) -> np.ndarray:
    """Averages a year of hourly data into one 24-value typical-day curve,
    normalized to [0, 1] so it can be scaled to any target peak wattage."""
    by_hour = hourly_df.groupby("hour")["P"].mean().reindex(range(24), fill_value=0.0)
    shape = by_hour.values.astype(float)
    max_val = shape.max()
    return shape / max_val if max_val > 0 else shape


def get_solar_profile(name: str, lat: float, lon: float, peak_watts: float,
                       year: int = 2019, use_cache: bool = True) -> np.ndarray:
    """Returns a 24-length array of green power (Watts) for one datacenter,
    scaled to `peak_watts`. Tries PVGIS + on-disk cache first; falls back to
    the Phase-1 synthetic generator if the API is unreachable."""
    cache_file = _cache_path(name)

    if use_cache and os.path.exists(cache_file):
        shape = pd.read_csv(cache_file)["shape"].values
        return shape * peak_watts

    try:
        hourly = fetch_pvgis_hourly(lat, lon, year=year)
        shape = typical_daily_shape(hourly)
        pd.DataFrame({"hour": range(24), "shape": shape}).to_csv(cache_file, index=False)
        return shape * peak_watts
    except Exception as e:
        print(f"[solar_data] WARNING: couldn't reach PVGIS for {name} ({e}). "
              f"Falling back to the Phase-1 synthetic solar curve for this datacenter. "
              f"This will resolve automatically once PVGIS is reachable (e.g. running "
              f"outside a restricted sandbox network).")
        synthetic = make_solar_profile(peak_watts, seed=hash(name) % (2**31))
        return synthetic


def build_real_solar_profiles(peak_watts_range=(800, 2500), seed: int = 0,
                               year: int = 2019, use_cache: bool = True):
    """Convenience function: returns a dict {dc_name: 24-length np.array}
    for all 10 locations in DATACENTER_LOCATIONS, each scaled to a random
    peak within `peak_watts_range` (same range Phase 1 used, so results stay
    comparable)."""
    rng = np.random.default_rng(seed)
    profiles = {}
    for name, lat, lon in DATACENTER_LOCATIONS:
        peak = rng.uniform(*peak_watts_range)
        profiles[name] = get_solar_profile(name, lat, lon, peak, year=year, use_cache=use_cache)
    return profiles
