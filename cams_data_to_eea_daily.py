"""
Daily EEA observations with CAMS dust averaged on each station's local day.

EEA daily timestamps are the station's local time (the ``Timezone`` column in
``DataExtract.csv``). CAMS time is UTC. This script builds the CAMS daily mean
over the local-day window of each station and writes a table the notebook can
load with the same columns as the hourly deduction.

Time-zone choice
----------------
The notebook converts EEA *hourly* timestamps with ``eea_to_utc``: every naive
stamp is treated as fixed UTC+1 (``Etc/GMT-1``, no daylight-saving time) and
then floored to a UTC day. That conversion is **not** applied here. Doing so
would shift a UTC+2 station's local day onto the wrong date.

Instead, the metadata offset (``UTC``, ``UTC+01``, ``UTC-04``, ...) is added to
each CAMS UTC timestamp and the mean is taken over the resulting local
calendar day::

    local day D  ==  UTC hours in [D 00:00 - offset, D+1 00:00 - offset)

The match to EEA is an exact local calendar date, not a nearest timestamp.
Offsets are the fixed labels EEA publishes. Daylight-saving transitions are
not applied, consistent with the hourly path's fixed UTC+1.

Memory
------
Hourly CAMS grids are never loaded in full. Each NetCDF file is opened lazily,
cut to the station bounding box and the dust variable, then read one UTC day
at a time, interpolated to station coordinates, and stored as float32.
Monthly station-hourly tables and the local-daily station table are written
under the same Google Drive folder the notebook uses
(``/content/drive/MyDrive/CAMS_Tool_output``) and reused when present.
"""

from __future__ import annotations

import calendar
import gc
import glob
import hashlib
import os
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

############### USER INPUT ####################
# Same knobs as Dust_discount_hourly.py / the notebook. Edit these to run
# this file as a script. Importing the module does not download or compute.
project_dir = "."
# project_dir = "/tsn.tno.nl/Data/SV/sv-059025_unix/ProjectData/EU/CAMS/C71/Werkdocumenten/wp-dust/"
dataset = "E2a"                 # E1a (verified) or E2a (up-to-date)
EEA_temporal_flag = "day"      # this script is the daily-observation path
POLLUTANT = "PM10"              # PM10 or PM2.5
YEAR = 2024
Countries = [
    "AD", "AL", "AT", "BA", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE",
    "ES", "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IS", "IT", "LT", "LU",
    "LV", "ME", "MK", "MT", "NL", "NO", "PL", "PT", "RO", "RS", "SE", "SI",
    "SK", "XK",
]
DOWNLOAD_EEA = False
DOWNLOAD_CAMS = False
COMPUTE_CAMS_DAILY = False      # True: rebuild local-daily means (monthly caches still reused). DOWNLOAD_CAMS forces this on.
USE_GOOGLE_DRIVE = False        # True: same Drive folder as the notebook
GDRIVE_OUTPUT_DIR = "/content/drive/MyDrive/CAMS_Tool_output"
VAR = "dust"
cams_dust_threshold = 5.0       # µg/m3
Station_Temporal_coverage = 65  # percent, notebook default
Basline_MA_days = 6             # notebook default (hourly script uses 15)
MIN_CAMS_HOURS = 18             # minimum equivalent hours inside a local day
CAMS_BUFFER_DEG = 0.2
###############################################

EEA_API_URL = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/"
EEA_DATASET_IDS = {"E2a": 1, "UTD": 1, "E1a": 2, "Historical": 3}
CAMS_DATASET_NAME = "cams-europe-air-quality-reanalyses"
POLLUTANT_THRESHOLDS = {"PM10": 50.0, "PM2.5": 25.0}


def eea_to_utc(series: pd.Series, source_tz: str = "CET") -> pd.Series:
    """Convert hourly EEA timestamps to UTC, assuming fixed UTC+1.

    This is the converter used by ``Dust_discount_hourly.py`` and by the
    notebook's hourly branch (naive stamps -> ``Etc/GMT-1`` -> UTC). It must
    not be used on EEA daily values: those stamps are already the station's
    local time, and a blanket UTC+1 shift would move stations whose metadata
    offset is not +1 onto the wrong day.
    """
    s = pd.to_datetime(series, errors="coerce")

    def _localize_if_naive(x):
        if pd.isna(x):
            return x
        return x.tz_localize("Etc/GMT-1") if x.tzinfo is None else x

    s = s.apply(_localize_if_naive)
    return s.dt.tz_convert("UTC")


def parse_tz_offset(tz_str) -> int:
    """Hours east of UTC from an EEA Timezone label.

    Handles the labels in ``DataExtract.csv``: ``UTC``, ``UTC+01``,
    ``UTC+02``, ``UTC-04``. Empty or unrecognised values return 0.
    ``UTC+01:00`` is accepted. The sign is kept (``UTC-04`` -> -4).
    """
    if tz_str is None or (isinstance(tz_str, float) and np.isnan(tz_str)):
        return 0
    try:
        if pd.isna(tz_str):
            return 0
    except (TypeError, ValueError):
        pass
    text = str(tz_str).strip().upper().replace(" ", "")
    if text in ("", "NAN", "NONE", "NAT"):
        return 0
    if "UTC" not in text:
        return 0
    rest = text.replace("UTC", "").replace("+", "")
    if rest in ("", "Z"):
        return 0
    if ":" in rest:
        rest = rest.split(":", 1)[0]
    return int(rest)


def naive_calendar_day(series: pd.Series) -> pd.Series:
    """Calendar date as naive midnight, without shifting the clock.

    ``filter_daily_by_coverage`` labels naive dates as UTC (``utc=True``),
    which does not change the calendar day. Dropping that timezone keeps the
    same date. Aware stamps are *not* converted to UTC first: EEA daily
    ``Start`` is local wall time, and converting it would change the day.
    """
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.floor("D")


def pollutant_daily_threshold(pollutant: str) -> float:
    try:
        return float(POLLUTANT_THRESHOLDS[pollutant])
    except KeyError as exc:
        raise ValueError(
            f"Pollutant {pollutant} not supported. Must be PM10 or PM2.5."
        ) from exc


def eea_request_body(
    countries: Sequence[str],
    pollutant: str,
    dataset_name: str,
    year: int,
    temporal_flag: str,
) -> dict:
    """JSON body of the EEA Parquet download, matching the notebook."""
    return {
        "countries": list(countries),
        "cities": [],
        "pollutants": [pollutant],
        "dataset": EEA_DATASET_IDS.get(dataset_name, 2),
        "dateTimeStart": f"{year - 1}-12-14T00:00:00Z",
        "dateTimeEnd": f"{year}-12-31T23:59:59Z",
        "aggregationType": temporal_flag,
        "email": "",
    }


def filter_daily_by_coverage(
    df_daily: pd.DataFrame,
    day_col: str = "day",
    station_col: str = "Samplingpoint",
    reference_year: int | None = None,
    min_pct: float = 75.0,
    keep_coverage_columns: bool = True,
) -> pd.DataFrame:
    """Keep every row for stations that meet ``min_pct`` coverage in ``reference_year``.

    Same rule as ``Dust_discount_hourly.py`` / ``util.filter_daily_by_coverage``:
    days outside the reference year stay, so the background window still has
    the days before 1 January. Naive daily dates are labelled UTC without
    moving the clock, so a local calendar day stays that calendar day.
    """
    df = df_daily.copy()
    df[day_col] = pd.to_datetime(df[day_col], utc=True, errors="coerce").dt.floor("D")
    df = df.dropna(subset=[day_col, station_col])
    df["year"] = df[day_col].dt.year

    coverage = (
        df.groupby([station_col, "year"])[day_col]
        .nunique()
        .rename("unique_days")
        .reset_index()
    )
    coverage["total_days"] = coverage["year"].apply(lambda y: 366 if calendar.isleap(y) else 365)
    coverage["coverage_percentage"] = (coverage["unique_days"] / coverage["total_days"]) * 100.0
    coverage["sufficient_coverage"] = coverage["coverage_percentage"] >= min_pct

    if reference_year is not None:
        qualifying = set(
            coverage.loc[
                (coverage["year"] == reference_year) & coverage["sufficient_coverage"],
                station_col,
            ].unique()
        )
    else:
        qualifying = set(coverage.loc[coverage["sufficient_coverage"], station_col].unique())

    out = df[df[station_col].isin(qualifying)].copy()
    if keep_coverage_columns:
        out = out.merge(coverage, on=[station_col, "year"], how="left", validate="many_to_one")
    return out


def compute_station_baseline(
    st_all: pd.DataFrame,
    st_year: pd.DataFrame,
    time_col: str = "day",
    value_col: str = "daily_mean",
    neighbor_n: int = 15,
) -> pd.Series:
    """Median of ``neighbor_n`` non-dust days before and after each dust exceedance.

    Same signature as ``Dust_discount_hourly.py``. The mask matches the notebook
    (``util.compute_station_baseline``): the background is computed only where
    ``dust_flag`` and ``Exceedance`` are both true. The hourly script currently
    has the exceedance term commented out.
    """
    mask_nondust = (
        (~st_all["dust_flag"].astype(bool))
        & st_all[time_col].notna()
        & st_all[value_col].notna()
    )
    nd_times = st_all.loc[mask_nondust, time_col].to_numpy()
    nd_vals = st_all.loc[mask_nondust, value_col].to_numpy()
    order = np.argsort(nd_times)
    nd_times = nd_times[order]
    nd_vals = nd_vals[order]

    mask_dust_exc = st_year["dust_flag"].astype(bool) & st_year["Exceedance"].astype(bool)
    dust_idx = st_year.index[mask_dust_exc]
    if dust_idx.empty or nd_times.size == 0:
        return pd.Series(index=st_year.index, dtype="float32")

    t_targets = st_year.loc[dust_idx, time_col].to_numpy()
    pos = np.searchsorted(nd_times, t_targets, side="left")
    medians = np.full(t_targets.shape[0], np.nan, dtype="float32")
    for i, p in enumerate(pos):
        before_vals = nd_vals[max(0, p - neighbor_n) : p]
        after_vals = nd_vals[p : min(nd_vals.shape[0], p + neighbor_n)]
        window = np.concatenate([before_vals, after_vals])
        if window.size > 0:
            medians[i] = np.nanmedian(window)

    result = pd.Series(index=st_year.index, dtype="float32")
    result.loc[dust_idx] = medians
    return result


def resolve_output_dirs(
    project_dir: str,
    use_google_drive: bool,
    gdrive_output_dir: str | None = None,
) -> tuple[str, bool]:
    """Return ``(persistent_dir, drive_is_active)``.

    Persistent files (EEA zip, CAMS zip, station caches, deduction Parquet)
    go to ``MyDrive/CAMS_Tool_output`` when Drive is on, otherwise to
    ``project_dir``. Extracted NetCDF stays under ``project_dir/IRA_dust``,
    which is where the notebook extracts them.
    """
    if not use_google_drive:
        return project_dir, False
    gdrive_output_dir = gdrive_output_dir or GDRIVE_OUTPUT_DIR
    if not os.path.exists("/content/drive"):
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except Exception as exc:
            print(f"Google Drive is not available ({exc}). Using {project_dir}.")
            return project_dir, False
    os.makedirs(gdrive_output_dir, exist_ok=True)
    print(f"Google Drive output directory: {gdrive_output_dir}")
    return gdrive_output_dir, True


def eea_zip_path(persistent_dir: str, pollutant: str, temporal_flag: str, dataset_name: str, year: int) -> str:
    """Zip path used by the notebook when ``USE_GOOGLE_DRIVE`` is True."""
    return os.path.join(
        persistent_dir,
        f"EEA_{pollutant}",
        temporal_flag,
        f"{dataset_name}_{pollutant}_{temporal_flag}_{year}.zip",
    )


def eea_extract_dir(project_dir: str, pollutant: str, year: int, dataset_name: str, temporal_flag: str) -> str:
    return os.path.join(project_dir, f"EEA_{pollutant}", str(year), dataset_name, temporal_flag)


def download_or_extract_eea(
    project_dir: str,
    persistent_dir: str,
    countries: Sequence[str],
    pollutant: str,
    dataset_name: str,
    year: int,
    temporal_flag: str,
    download: bool,
) -> str:
    """Download EEA Parquet the way the notebook does, or reuse the saved zip.

    API, dataset ids (E2a=1, E1a=2), country list, pollutant, year span
    (14 December of the previous year through 31 December) and
    ``aggregationType`` match the notebook. When ``download`` is False the
    existing zip is extracted. When it is True and the zip is already there,
    the notebook replaces it; this function does the same.
    """
    import requests
    import zipfile
    from datetime import datetime

    extract_dir = eea_extract_dir(project_dir, pollutant, year, dataset_name, temporal_flag)
    os.makedirs(extract_dir, exist_ok=True)
    zip_target = eea_zip_path(persistent_dir, pollutant, temporal_flag, dataset_name, year)
    os.makedirs(os.path.dirname(zip_target), exist_ok=True)

    if download:
        if os.path.exists(zip_target):
            os.remove(zip_target)
            print(f"Removed existing EEA data from {zip_target}. Proceeding with new download.")
        body = eea_request_body(countries, pollutant, dataset_name, year, temporal_flag)
        response = requests.post(f"{EEA_API_URL}ParquetFile/async", json=body, timeout=120)
        response.raise_for_status()
        download_url = response.text.strip()
        print(f"Requesting download from: {download_url}")
        t_start = datetime.now()
        parquet_response = None
        while True:
            if (datetime.now() - t_start).total_seconds() > 3600:
                break
            parquet_response = requests.get(download_url, timeout=120)
            if parquet_response.status_code == 404:
                time.sleep(20)
            else:
                break
        if parquet_response is None or parquet_response.status_code != 200:
            raise RuntimeError(
                f"EEA download did not become ready (last status "
                f"{getattr(parquet_response, 'status_code', None)})."
            )
        with open(zip_target, "wb") as fp:
            fp.write(parquet_response.content)
        print(f"Download Successful: {zip_target}")
    elif not os.path.exists(zip_target):
        print(
            f"DOWNLOAD_EEA is False, but no existing zip file was found at {zip_target}. "
            "Please set DOWNLOAD_EEA to True."
        )
        return extract_dir

    if os.path.exists(zip_target):
        import zipfile

        print(f"Extracting EEA zip: {zip_target}")
        with zipfile.ZipFile(zip_target, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"EEA data extracted to {extract_dir}")
    return extract_dir


def cams_download_plan(year: int) -> list[dict]:
    """Five CAMS requests: December of the previous year, then four quarters."""
    return [
        {"filename": f"CAMS_IRA_{year - 1}_12.zip", "year": [f"{year - 1}"], "month": ["12"]},
        {"filename": f"CAMS_IRA_{year}_q1.zip", "year": [f"{year}"], "month": ["01", "02", "03"]},
        {"filename": f"CAMS_IRA_{year}_q2.zip", "year": [f"{year}"], "month": ["04", "05", "06"]},
        {"filename": f"CAMS_IRA_{year}_q3.zip", "year": [f"{year}"], "month": ["07", "08", "09"]},
        {"filename": f"CAMS_IRA_{year}_q4.zip", "year": [f"{year}"], "month": ["10", "11", "12"]},
    ]


def resolve_run_flags(
    *,
    download_eea: bool,
    download_cams: bool,
    compute_cams_daily: bool,
    load_parquet: bool,
    eea_temporal_flag: str,
) -> dict:
    """Collapse the notebook switches that otherwise contradict each other.

    ``DOWNLOAD_CAMS`` means new hourly CAMS files, so the daily product has to
    be rebuilt (``compute_cams_daily``). How that average is taken follows
    ``eea_temporal_flag``: ``hour`` is a UTC day, ``day`` is each station's
    local day. A new download or a new daily product cannot be skipped by
    loading the previous deduction Parquet.
    """
    if eea_temporal_flag not in ("hour", "day"):
        raise ValueError("EEA_temporal_flag must be 'hour' or 'day'.")
    notes: list[str] = []
    if download_cams and not compute_cams_daily:
        compute_cams_daily = True
        notes.append("DOWNLOAD_CAMS is True, so COMPUTE_CAMS_DAILY has been set to True.")
    if (download_eea or download_cams or compute_cams_daily) and load_parquet:
        load_parquet = False
        notes.append(
            "LOAD_PARQUET_DATA has been set to False so the new download or daily CAMS product is used."
        )
    return {
        "download_eea": bool(download_eea),
        "download_cams": bool(download_cams),
        "compute_cams_daily": bool(compute_cams_daily),
        "load_parquet": bool(load_parquet),
        "eea_temporal_flag": eea_temporal_flag,
        "local_day": eea_temporal_flag == "day",
        "notes": notes,
    }


def stored_cams_zip(persistent_dir: str, project_dir: str, var: str, filename: str) -> str:
    """Path of a CAMS zip in the output folder (Google Drive when that is on).

    A zip that exists only next to the extracted NetCDF is copied into the
    output folder. The hourly NetCDF is not copied there.
    """
    import shutil

    zip_dir = os.path.join(persistent_dir, f"IRA_{var}")
    os.makedirs(zip_dir, exist_ok=True)
    stored = os.path.join(zip_dir, filename)
    local = os.path.join(project_dir, f"IRA_{var}", filename)
    if (
        not os.path.isfile(stored)
        and os.path.isfile(local)
        and os.path.abspath(stored) != os.path.abspath(local)
    ):
        print(f"Saving CAMS zip to the output folder: {stored}")
        shutil.copy2(local, stored)
    return stored


def download_and_extract_cams(
    project_dir: str,
    persistent_dir: str,
    year: int,
    var: str,
    download: bool,
    extract: bool = True,
) -> str:
    """Download CAMS zips into the output folder and optionally extract them.

    The zip is the stored file (Google Drive when ``persistent_dir`` is the
    Drive output folder). Hourly NetCDF is extracted to ``project_dir/IRA_{var}``
    only when ``extract`` is true, which is the daily-mean step of a run.
    Extracted NetCDF is not written back to Drive.
    """
    import zipfile

    extract_dir = os.path.join(project_dir, f"IRA_{var}")
    os.makedirs(extract_dir, exist_ok=True)
    zip_dir = os.path.join(persistent_dir, f"IRA_{var}")
    print(f"CAMS zip folder: {zip_dir}")
    if extract:
        print(f"Hourly NetCDF for this run only: {extract_dir}")

    client = None
    for config in cams_download_plan(year):
        zip_file_path = stored_cams_zip(persistent_dir, project_dir, var, config["filename"])
        if download and not os.path.exists(zip_file_path):
            import cdsapi

            if client is None:
                client = cdsapi.Client()
            print(f"Downloading CAMS zip to {zip_file_path}...")
            request = {
                "variable": [var],
                "model": ["ensemble"],
                "level": ["0"],
                "type": ["interim_reanalysis"],
                "year": config["year"],
                "month": config["month"],
            }
            client.retrieve(CAMS_DATASET_NAME, request).download(zip_file_path)
            print(f"Download Successful: {config['filename']}")

        if not os.path.exists(zip_file_path):
            print(f"CAMS zip not found: {zip_file_path}. Set DOWNLOAD_CAMS=True to fetch it.")
            continue
        if not extract:
            print(f"Keeping zip {config['filename']}. Not extracted in this run.")
            continue
        print(f"Extracting {zip_file_path} for this run...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted to {extract_dir}. The zip stays in the output folder.")
    return extract_dir


def load_and_filter_eea_daily(folder: str, temporal_flag: str = "day") -> pd.DataFrame:
    """Load EEA daily Parquet into the hourly script's daily table shape.

    Columns on return: ``Pollutant``, ``day``, ``Samplingpoint``, ``daily_mean``,
    ``unit``. ``day`` is the naive local calendar date of ``Start``. Validity
    1, verification below 3, non-negative values, and the requested aggregation
    are kept. Timestamps are not converted to UTC.
    """
    files = sorted(glob.glob(os.path.join(folder, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No EEA parquet files under {folder}. Download them first "
            "(DOWNLOAD_EEA = True) or point project_dir at an existing extract."
        )
    needed = ["Value", "Validity", "Verification", "AggType", "Start", "Samplingpoint", "Unit", "Pollutant"]
    frames = []
    for file in files:
        tmp = pd.read_parquet(file)
        missing = [col for col in needed if col not in tmp.columns]
        if missing:
            raise KeyError(f"{file} is missing columns {missing}")
        tmp = tmp.loc[:, needed]
        tmp["Value"] = pd.to_numeric(tmp["Value"], errors="coerce").astype("float32")
        frames.append(tmp)
    obs = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    obs["Validity"] = pd.to_numeric(obs["Validity"], errors="coerce")
    obs["Verification"] = pd.to_numeric(obs["Verification"], errors="coerce")
    obs["AggType"] = obs["AggType"].astype(str).str.lower()
    mask = (
        obs["Validity"].eq(1)
        & obs["Verification"].lt(3)
        & obs["AggType"].eq(temporal_flag.lower())
        & (obs["Value"] >= 0)
    )
    obs = obs.loc[mask]
    # Local wall time. Floor to the calendar date and drop any tz without converting.
    obs["day"] = naive_calendar_day(obs["Start"])
    obs = obs.dropna(subset=["day", "Samplingpoint"])
    obs = obs.drop_duplicates(subset=["Pollutant", "Samplingpoint", "day"])
    daily = (
        obs.groupby(["Pollutant", "day", "Samplingpoint"], sort=True, observed=False)
        .agg(daily_mean=("Value", "mean"), unit=("Unit", "first"))
        .reset_index()
    )
    daily["daily_mean"] = daily["daily_mean"].astype("float32")
    print(
        f"EEA daily rows after validity filter: {len(daily):,} "
        f"({daily['Samplingpoint'].nunique()} stations)"
    )
    del obs
    gc.collect()
    return daily


def merge_station_metadata(
    df: pd.DataFrame,
    metadata_path: str,
    pollutant: str | None = None,
) -> pd.DataFrame:
    """Attach coordinates and the fixed UTC offset. Join key matches the notebook."""
    metadata = pd.read_csv(metadata_path, low_memory=False)
    metadata["Samplingpoint"] = (
        metadata["Air Quality Station EoI Code"].astype(str).str[:2]
        + "/"
        + metadata["Sampling Point Id"].astype(str)
    )
    if pollutant and "Air Pollutant" in metadata.columns:
        subset = metadata[metadata["Air Pollutant"] == pollutant]
        if subset.empty:
            print(
                f"Warning: no metadata rows for pollutant {pollutant}; "
                "using every row in DataExtract.csv."
            )
        else:
            metadata = subset
    metadata["tz_offset"] = metadata["Timezone"].map(parse_tz_offset).astype("int16")
    stations = metadata[
        ["Samplingpoint", "Longitude", "Latitude", "Altitude", "Timezone", "tz_offset"]
    ].drop_duplicates(subset=["Samplingpoint"], keep="first")
    out = df.merge(stations, on="Samplingpoint", how="left")
    missing = out["Latitude"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows have no coordinates in {metadata_path}.")
    unknown_tz = out["Timezone"].isna().sum()
    if unknown_tz:
        print(
            f"Warning: {unknown_tz} rows have no Timezone label; "
            "their CAMS window falls back to UTC (offset 0)."
        )
    out["tz_offset"] = out["tz_offset"].fillna(0).astype("int16")
    out["Longitude"] = pd.to_numeric(out["Longitude"], errors="coerce")
    out["Latitude"] = pd.to_numeric(out["Latitude"], errors="coerce")
    out["Altitude"] = pd.to_numeric(out["Altitude"], errors="coerce")
    return out


def _dim_name(dims: Iterable[str], prefix: str) -> str:
    for dim in dims:
        if str(dim).lower().startswith(prefix):
            return str(dim)
    raise KeyError(f"No dimension starting with '{prefix}' in {list(dims)}")


def _index_window(values, low: float, high: float, pad: float) -> slice:
    """Index slice covering ``[low - pad, high + pad]``, plus one grid cell each side.

    A pad smaller than the grid spacing must still keep the cells that bracket
    the stations, otherwise linear interpolation is asked to run on an empty
    array. Works for ascending and descending coordinates.
    """
    values = np.asarray(values, dtype="float64")
    if values.size == 0:
        return slice(0, 0)
    inside = np.flatnonzero((values >= low - pad) & (values <= high + pad))
    if inside.size == 0:
        mid = 0.5 * (low + high)
        inside = np.array([int(np.argmin(np.abs(values - mid)))])
    start = max(0, int(inside.min()) - 1)
    stop = min(int(values.size), int(inside.max()) + 2)
    return slice(start, stop)


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def list_cams_netcdf(project_dir: str, persistent_dir: str, var: str, year: int) -> list[str]:
    """NetCDF files for ``year`` plus December of the previous year.

    December is required because a positive local offset reaches back into
    31 December, and the EEA download itself starts on 14 December so the
    baseline window is defined. The hourly notebook glob only contains
    ``year``; the daily path includes that extra month on purpose.
    """
    patterns = [
        os.path.join(project_dir, f"IRA_{var}", "cams.eaq.ira*.nc"),
        os.path.join(persistent_dir, f"IRA_{var}", "cams.eaq.ira*.nc"),
        os.path.join(project_dir, "IRA_dust", "cams.eaq.ira*.nc"),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    selected = []
    year_token = str(year)
    december_token = f"{year - 1}-12"
    for path in sorted(set(found)):
        name = os.path.basename(path)
        if var not in name and "dust" not in name.lower():
            continue
        if year_token in name or december_token in name:
            selected.append(path)
    return selected


def read_cams_domain(path: str, var_name: str = "dust") -> dict:
    """Lat/lon bounds and names from coordinate arrays only (not the grid)."""
    ds = xr.open_dataset(path, chunks={"time": 24})
    try:
        name = _resolve_var_name(ds, var_name)
        da = ds[name]
        lat_name = _dim_name(da.dims, "lat")
        lon_name = _dim_name(da.dims, "lon")
        time_name = _dim_name(da.dims, "time")
        lat = np.asarray(da[lat_name].values, dtype="float64")
        lon = np.asarray(da[lon_name].values, dtype="float64")
        return {
            "var_name": name,
            "lat_name": lat_name,
            "lon_name": lon_name,
            "time_name": time_name,
            "lat_min": float(np.nanmin(lat)),
            "lat_max": float(np.nanmax(lat)),
            "lon_min": float(np.nanmin(lon)),
            "lon_max": float(np.nanmax(lon)),
        }
    finally:
        ds.close()


def _resolve_var_name(ds: xr.Dataset, var_name: str) -> str:
    if var_name in ds.data_vars:
        return var_name
    matches = [name for name in ds.data_vars if var_name.lower() in name.lower()]
    if len(matches) == 1:
        return matches[0]
    if "dust" in ds.data_vars:
        return "dust"
    dust_like = [name for name in ds.data_vars if "dust" in name.lower()]
    if len(dust_like) == 1:
        return dust_like[0]
    raise KeyError(
        f"Variable '{var_name}' not found. Data variables: {list(ds.data_vars)}"
    )


def station_cache_tag(stations: pd.DataFrame, station_col: str = "Samplingpoint") -> str:
    key = stations[[station_col, "Latitude", "Longitude", "tz_offset"]].copy()
    key["Latitude"] = pd.to_numeric(key["Latitude"]).round(5)
    key["Longitude"] = pd.to_numeric(key["Longitude"]).round(5)
    key["tz_offset"] = key["tz_offset"].astype(int)
    key = key.drop_duplicates(station_col).sort_values(station_col)
    payload = key.to_csv(index=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def _iter_utc_days(times: np.ndarray):
    """Yield ``(day, integer positions)`` for each UTC calendar day, in order."""
    stamps = pd.to_datetime(times)
    order = np.argsort(stamps.to_numpy())
    ordered = stamps.to_numpy()[order]
    days = pd.to_datetime(ordered).floor("D").to_numpy()
    if len(days) == 0:
        return
    change = np.flatnonzero(days[1:] != days[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(days)]
    for start, end in zip(starts, ends):
        yield pd.Timestamp(days[start]), order[start:end]


def minimum_samples_for_day(step_hours: float, min_hours: int) -> int:
    """How many valid samples a local day needs.

    ``min_hours`` is defined on a 24-hour clock (default 18, i.e. 75%).
    A 3-hourly file has 8 samples per day, so the same fraction is 6.
    """
    step = max(float(step_hours), 1.0)
    full_day = max(1, int(round(24.0 / step)))
    return max(1, int(np.ceil(full_day * (min_hours / 24.0))))


def _median_step_hours(times: np.ndarray) -> float:
    if len(times) < 2:
        return 1.0
    stamps = pd.to_datetime(times).sort_values().to_numpy(dtype="datetime64[ns]")
    delta_hours = np.diff(stamps).astype("timedelta64[s]").astype(np.float64) / 3600.0
    delta_hours = delta_hours[np.isfinite(delta_hours) & (delta_hours > 0)]
    if delta_hours.size == 0:
        return 1.0
    return float(np.median(delta_hours))


def extract_hourly_at_stations(
    nc_path: str,
    stations: pd.DataFrame,
    var_name: str = "dust",
    buffer: float = 0.2,
    station_col: str = "Samplingpoint",
) -> pd.DataFrame:
    """Interpolate one CAMS file to stations, one UTC day at a time.

    Returns a wide table: column ``time`` (naive UTC) plus one float32 column
    per sampling point. The full hourly grid is not loaded. After each day the
    day-sized array is deleted.
    """
    ds = xr.open_dataset(nc_path, chunks={"time": 24})
    pieces = []
    try:
        resolved = _resolve_var_name(ds, var_name)
        da = ds[resolved]
        time_name = _dim_name(da.dims, "time")
        lat_name = _dim_name(da.dims, "lat")
        lon_name = _dim_name(da.dims, "lon")
        if da[lat_name].ndim != 1 or da[lon_name].ndim != 1:
            raise ValueError("CAMS latitude/longitude are expected to be 1-D coordinates.")

        lat_min = float(stations["Latitude"].min())
        lat_max = float(stations["Latitude"].max())
        lon_min = float(stations["Longitude"].min())
        lon_max = float(stations["Longitude"].max())
        da = da.isel(
            {
                lat_name: _index_window(da[lat_name].values, lat_min, lat_max, buffer),
                lon_name: _index_window(da[lon_name].values, lon_min, lon_max, buffer),
            }
        )
        ids = stations[station_col].astype(str).to_numpy()
        lat_da = xr.DataArray(stations["Latitude"].to_numpy(dtype="float64"), dims="station")
        lon_da = xr.DataArray(stations["Longitude"].to_numpy(dtype="float64"), dims="station")
        times = pd.to_datetime(da[time_name].values)
        n_days = 0
        for _day, positions in _iter_utc_days(times.to_numpy()):
            sub = da.isel({time_name: np.asarray(positions, dtype=int)})
            if sub.dtype != np.float32:
                sub = sub.astype("float32")
            loaded = sub.load()
            interpolated = loaded.interp({lat_name: lat_da, lon_name: lon_da}, method="linear")
            interpolated = interpolated.transpose(time_name, "station")
            values = np.asarray(interpolated.values, dtype="float32")
            if np.isnan(values).any():
                nearest = loaded.interp(
                    {lat_name: lat_da, lon_name: lon_da}, method="nearest"
                ).transpose(time_name, "station")
                nearest_values = np.asarray(nearest.values, dtype="float32")
                values = np.where(np.isnan(values), nearest_values, values)
                del nearest, nearest_values
            block = pd.DataFrame(values, columns=ids)
            block.insert(0, "time", pd.to_datetime(loaded[time_name].values))
            pieces.append(block)
            n_days += 1
            del loaded, interpolated, values, sub, block
            gc.collect()
        print(
            f"  {os.path.basename(nc_path)}: {n_days} UTC days, "
            f"{len(ids)} stations, rss={_rss_mb()}"
        )
    finally:
        ds.close()
        gc.collect()
    if not pieces:
        return pd.DataFrame(columns=["time"])
    hourly = pd.concat(pieces, ignore_index=True)
    del pieces
    gc.collect()
    return _as_utc_naive_time(hourly)


def _as_utc_naive_time(hourly: pd.DataFrame) -> pd.DataFrame:
    """CAMS timestamps as naive UTC. Aware values are converted to UTC first."""
    stamps = pd.to_datetime(hourly["time"])
    if getattr(stamps.dt, "tz", None) is not None:
        stamps = stamps.dt.tz_convert("UTC").dt.tz_localize(None)
    hourly = hourly.copy()
    hourly["time"] = stamps
    hourly = hourly.drop_duplicates(subset=["time"], keep="first")
    return hourly.sort_values("time").reset_index(drop=True)


def aggregate_local_daily(
    hourly: pd.DataFrame,
    tz_offset: pd.Series,
    min_hours: int = MIN_CAMS_HOURS,
    station_col: str = "Samplingpoint",
) -> pd.DataFrame:
    """Mean CAMS dust on each station's local calendar day.

    ``hourly`` is wide: ``time`` in UTC plus one column per station.
    Stations that share an offset are shifted together. A local day is kept
    when the number of finite samples reaches the 75% rule in
    ``minimum_samples_for_day``. Missing days are absent (callers left-join,
    so they become NaN rather than a nearest-day fill).
    """
    if hourly.empty or "time" not in hourly.columns:
        return pd.DataFrame(columns=[station_col, "day", "cams_dust", "n_cams_hours"])
    # One row per UTC timestamp. Overlapping monthly files must not be averaged twice.
    hourly = _as_utc_naive_time(hourly)
    time_utc = hourly["time"]
    step_hours = _median_step_hours(time_utc.to_numpy())
    min_count = minimum_samples_for_day(step_hours, min_hours)
    pieces = []
    offsets = tz_offset.copy()
    offsets.index = offsets.index.astype(str)
    for offset, group in offsets.groupby(offsets):
        cols = [col for col in group.index if col in hourly.columns]
        if not cols:
            continue
        # Local clock = UTC + offset. Positive offsets (CET, EET) move the
        # label forward; negative offsets move it backward.
        local_time = time_utc + pd.to_timedelta(int(offset), unit="h")
        local_day = local_time.dt.floor("D")
        values = hourly[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
        block = pd.DataFrame(values, columns=cols)
        block["day"] = local_day.to_numpy()
        grouped = block.groupby("day", sort=True)
        means = grouped.mean()
        counts = grouped.count()
        keep = counts >= min_count
        means = means.where(keep)
        n_day, n_station = means.shape
        if n_day == 0 or n_station == 0:
            continue
        station_ids = np.asarray(means.columns, dtype=object)
        piece = pd.DataFrame(
            {
                "day": np.repeat(means.index.to_numpy(), n_station),
                station_col: np.tile(station_ids, n_day),
                "cams_dust": means.to_numpy(dtype="float32").reshape(-1),
                "n_cams_hours": counts.to_numpy(dtype="int16").reshape(-1),
            }
        )
        piece = piece[piece["n_cams_hours"] >= min_count]
        pieces.append(piece)
        del block, values, means, counts, piece
    gc.collect()
    if not pieces:
        return pd.DataFrame(columns=[station_col, "day", "cams_dust", "n_cams_hours"])
    out = pd.concat(pieces, ignore_index=True)
    out["day"] = pd.to_datetime(out["day"])
    out["cams_dust"] = out["cams_dust"].astype("float32")
    return out


def _hourly_wide_from_station_dataset(ds: xr.Dataset, var_name: str, station_col: str) -> pd.DataFrame:
    """Hourly values already reduced to a station dimension (tests / small inputs)."""
    resolved = _resolve_var_name(ds, var_name)
    da = ds[resolved]
    time_name = _dim_name(da.dims, "time")
    if "station" not in da.dims:
        raise KeyError("Expected a 'station' dimension on a pre-interpolated dataset.")
    loaded = da.astype("float32").load()
    ids = np.asarray(loaded["station"].values).astype(str)
    values = np.asarray(loaded.values, dtype="float32")
    # (time, station) required
    if loaded.dims[0] != time_name:
        values = np.moveaxis(values, loaded.dims.index(time_name), 0)
    frame = pd.DataFrame(values, columns=ids)
    frame.insert(0, "time", pd.to_datetime(loaded[time_name].values))
    del loaded
    gc.collect()
    return frame


def _gridded_dataset_to_hourly_wide(
    ds: xr.Dataset,
    stations: pd.DataFrame,
    var_name: str,
    buffer: float,
    station_col: str,
) -> pd.DataFrame:
    """Same day-by-day interpolation as the file reader, for an open dataset."""
    resolved = _resolve_var_name(ds, var_name)
    da = ds[resolved]
    time_name = _dim_name(da.dims, "time")
    lat_name = _dim_name(da.dims, "lat")
    lon_name = _dim_name(da.dims, "lon")
    lat_min = float(stations["Latitude"].min())
    lat_max = float(stations["Latitude"].max())
    lon_min = float(stations["Longitude"].min())
    lon_max = float(stations["Longitude"].max())
    da = da.isel(
        {
            lat_name: _index_window(da[lat_name].values, lat_min, lat_max, buffer),
            lon_name: _index_window(da[lon_name].values, lon_min, lon_max, buffer),
        }
    )
    ids = stations[station_col].astype(str).to_numpy()
    lat_da = xr.DataArray(stations["Latitude"].to_numpy(dtype="float64"), dims="station")
    lon_da = xr.DataArray(stations["Longitude"].to_numpy(dtype="float64"), dims="station")
    times = pd.to_datetime(da[time_name].values)
    pieces = []
    for _day, positions in _iter_utc_days(times.to_numpy()):
        sub = da.isel({time_name: np.asarray(positions, dtype=int)}).astype("float32")
        loaded = sub.load()
        interpolated = loaded.interp({lat_name: lat_da, lon_name: lon_da}, method="linear")
        interpolated = interpolated.transpose(time_name, "station")
        values = np.asarray(interpolated.values, dtype="float32")
        if np.isnan(values).any():
            nearest = loaded.interp(
                {lat_name: lat_da, lon_name: lon_da}, method="nearest"
            ).transpose(time_name, "station")
            values = np.where(np.isnan(values), np.asarray(nearest.values, dtype="float32"), values)
            del nearest
        block = pd.DataFrame(values, columns=ids)
        block.insert(0, "time", pd.to_datetime(loaded[time_name].values))
        pieces.append(block)
        del loaded, interpolated, values, sub, block
        gc.collect()
    if not pieces:
        return pd.DataFrame(columns=["time"])
    return pd.concat(pieces, ignore_index=True)


def _cache_paths(cache_dir: str, tag: str, year: int, min_hours: int, source_name: str | None = None) -> str:
    folder = os.path.join(cache_dir, "IRA_dust")
    os.makedirs(folder, exist_ok=True)
    if source_name is None:
        return os.path.join(folder, f"cams_dust_local_daily_stations_{tag}_{year}_h{min_hours}.parquet")
    safe = os.path.splitext(os.path.basename(source_name))[0]
    return os.path.join(folder, f"cams_dust_station_hourly_{tag}_{safe}.parquet")


def _load_or_extract_hourly(
    paths: Sequence[str],
    stations: pd.DataFrame,
    var_name: str,
    buffer: float,
    cache_dir: str | None,
    tag: str,
    station_col: str,
) -> pd.DataFrame:
    frames = []
    for path in paths:
        cache_file = None
        if cache_dir:
            cache_file = _cache_paths(cache_dir, tag, year=0, min_hours=0, source_name=path)
            if os.path.exists(cache_file):
                print(f"Reusing station-hourly cache {cache_file}")
                frames.append(pd.read_parquet(cache_file))
                continue
        print(f"Extracting station locations from {path}")
        hourly = extract_hourly_at_stations(
            path, stations, var_name=var_name, buffer=buffer, station_col=station_col
        )
        if cache_file:
            hourly.to_parquet(cache_file, index=False, compression="snappy")
            print(f"Saved station-hourly cache {cache_file}")
        frames.append(hourly)
        del hourly
        gc.collect()
    if not frames:
        return pd.DataFrame(columns=["time"])
    return pd.concat(frames, ignore_index=True)


def add_cams_local_daily_dust(
    cams_ds,
    df: pd.DataFrame,
    var_name: str = "dust",
    buffer: float = 0.2,
    cache_dir: str | None = None,
    reuse_cache: bool = True,
    recompute_daily: bool = False,
    min_hours: int = MIN_CAMS_HOURS,
    time_col: str = "day",
    station_col: str = "Samplingpoint",
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    tz_col: str = "Timezone",
    year: int | None = None,
) -> pd.DataFrame:
    """Attach ``cams_dust`` averaged on each station's local day.

    Parameters
    ----------
    cams_ds
        NetCDF path, list of paths, a glob string, or an ``xarray.Dataset``.
        A dataset with lat/lon is interpolated one UTC day at a time. A dataset
        that already has a ``station`` dimension is treated as hourly UTC at
        those stations. File inputs are the Colab path: grids stay on disk.
    df
        Daily EEA table with ``time_col``, coordinates, and either ``tz_offset``
        or ``tz_col`` (EEA labels such as ``UTC+01``).
    cache_dir
        Directory for intermediate Parquet files. The notebook passes the
        Google Drive output folder. Files already present are reused.
    recompute_daily
        When True, rebuild the local-daily table. Monthly station-hourly
        caches are still reused so the grids are not read again.
    min_hours
        Minimum equivalent hours (out of 24) required to keep a local day.

    Returns
    -------
    DataFrame
        Copy of the in-domain rows of ``df`` with ``cams_dust`` (float32) and
        ``n_cams_hours``. The join is an exact local date.
    """
    out = df.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    if "tz_offset" not in out.columns:
        if tz_col not in out.columns:
            raise KeyError(
                "Station timezone is required. Merge DataExtract.csv so the "
                "table has 'Timezone' or 'tz_offset'."
            )
        out["tz_offset"] = out[tz_col].map(parse_tz_offset)
    out["tz_offset"] = out["tz_offset"].fillna(0).astype("int16")
    out["_local_day"] = naive_calendar_day(out[time_col])

    paths: list[str] | None = None
    dataset = cams_ds if isinstance(cams_ds, xr.Dataset) else None
    if dataset is None:
        if isinstance(cams_ds, (str, os.PathLike)):
            text = str(cams_ds)
            paths = sorted(glob.glob(text)) if any(ch in text for ch in "*?[") else [text]
        else:
            paths = [str(path) for path in cams_ds]
        paths = [path for path in paths if os.path.isfile(path)]

    domain = None
    if dataset is not None and any(str(dim).lower().startswith("lat") for dim in dataset.dims):
        lat_name = _dim_name(dataset.dims, "lat")
        lon_name = _dim_name(dataset.dims, "lon")
        domain = {
            "lat_min": float(dataset[lat_name].min()),
            "lat_max": float(dataset[lat_name].max()),
            "lon_min": float(dataset[lon_name].min()),
            "lon_max": float(dataset[lon_name].max()),
        }
    elif paths:
        domain = read_cams_domain(paths[0], var_name)

    if domain is not None:
        n_before = int(out[station_col].nunique())
        lon = out[lon_col]
        if domain["lon_min"] >= 0:
            lon = lon.where(lon >= 0, lon + 360.0)
        inside_mask = (
            out[lat_col].between(domain["lat_min"], domain["lat_max"])
            & lon.between(domain["lon_min"], domain["lon_max"])
        )
        out = out.loc[inside_mask].copy()
        print(
            f"Stations filtered: {n_before - out[station_col].nunique()} removed, "
            f"{out[station_col].nunique()} remain within CAMS domain."
        )

    stations = (
        out[[station_col, lat_col, lon_col, "tz_offset"]]
        .dropna(subset=[station_col, lat_col, lon_col])
        .drop_duplicates(station_col, keep="first")
        .rename(columns={lat_col: "Latitude", lon_col: "Longitude"})
    )
    if domain is not None and domain["lon_min"] >= 0:
        shifted = stations["Longitude"].where(stations["Longitude"] >= 0, stations["Longitude"] + 360.0)
        stations = stations.copy()
        stations["Longitude"] = shifted
    if stations.empty:
        out["cams_dust"] = np.float32(np.nan)
        out["n_cams_hours"] = np.int16(0)
        return out.drop(columns=["_local_day"])

    stations[station_col] = stations[station_col].astype(str)
    tag = station_cache_tag(stations, station_col)
    target_year = int(year) if year is not None else int(out["_local_day"].dt.year.min())
    daily_cache = (
        _cache_paths(cache_dir, tag, target_year, min_hours) if cache_dir else None
    )
    if (
        cache_dir
        and reuse_cache
        and not recompute_daily
        and daily_cache
        and not os.path.exists(daily_cache)
        and not paths
        and dataset is None
    ):
        # Grids are gone but a previous run may have left a station table behind.
        pattern = os.path.join(
            cache_dir,
            "IRA_dust",
            f"cams_dust_local_daily_stations_*_{target_year}_h{min_hours}.parquet",
        )
        found = sorted(glob.glob(pattern))
        if len(found) == 1:
            print(f"CAMS grids are not on disk. Reusing {found[0]}")
            daily_cache = found[0]

    daily = None
    if daily_cache and reuse_cache and not recompute_daily and os.path.exists(daily_cache):
        print(f"Reusing local-daily CAMS cache {daily_cache}")
        daily = pd.read_parquet(daily_cache)
    else:
        if dataset is not None:
            if any(str(dim).lower().startswith("lat") for dim in dataset.dims):
                hourly = _gridded_dataset_to_hourly_wide(
                    dataset, stations, var_name, buffer, station_col
                )
            else:
                hourly = _hourly_wide_from_station_dataset(dataset, var_name, station_col)
        else:
            if not paths:
                raise FileNotFoundError(
                    "No CAMS NetCDF files found and no local-daily cache to reuse. "
                    "Set DOWNLOAD_CAMS and COMPUTE_CAMS_DAILY, or place "
                    "cams.eaq.ira*.nc under IRA_dust/."
                )
            hourly = _load_or_extract_hourly(
                paths, stations, var_name, buffer, cache_dir, tag, station_col
            )
        tz = stations.set_index(station_col)["tz_offset"]
        print(
            f"Temporal local-time resampling for {len(stations)} stations "
            f"(step-aware minimum of {min_hours}/24 h)..."
        )
        daily = aggregate_local_daily(hourly, tz, min_hours=min_hours, station_col=station_col)
        del hourly
        gc.collect()
        if daily_cache:
            daily.to_parquet(daily_cache, index=False, compression="snappy")
            print(f"Saved local-daily CAMS cache {daily_cache}")

    daily = daily.copy()
    daily[station_col] = daily[station_col].astype(str)
    daily["_cams_day"] = naive_calendar_day(daily["day"])
    daily = daily.sort_values("n_cams_hours", ascending=False)
    daily = daily.drop_duplicates(subset=[station_col, "_cams_day"], keep="first")
    out[station_col] = out[station_col].astype(str)
    # Exact local calendar date. A missing CAMS day stays NaN (no nearest-day fill).
    merged = out.merge(
        daily[[station_col, "_cams_day", "cams_dust", "n_cams_hours"]],
        left_on=[station_col, "_local_day"],
        right_on=[station_col, "_cams_day"],
        how="left",
    )
    merged["cams_dust"] = merged["cams_dust"].astype("float32")
    merged["n_cams_hours"] = merged["n_cams_hours"].fillna(0).astype("int16")
    merged = merged.drop(columns=["_local_day", "_cams_day"])
    return merged


def run_deduction_for_pollutant(
    df: pd.DataFrame,
    year: int,
    pollutant: str,
    dust_threshold: float,
    neighbor_n: int,
    time_col: str = "day",
    value_col: str = "daily_mean",
) -> pd.DataFrame:
    """Full deduction, then keep the target year. Mirrors the notebook cells."""
    work = df.copy()
    work[time_col] = naive_calendar_day(work[time_col])
    limit = pollutant_daily_threshold(pollutant)
    work["dust_flag"] = work["cams_dust"] > dust_threshold
    work["Exceedance"] = work[value_col] > limit
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").astype("float32")
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce").dt.floor("D")

    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    target = work[(work[time_col] >= start) & (work[time_col] < end)].copy()
    if target.empty and not work.empty:
        available = int(work[time_col].dt.year.dropna().iloc[0])
        print(f"Warning: No data for {year}. Using available data from {available}")
        start = pd.Timestamp(f"{available}-01-01", tz="UTC")
        end = pd.Timestamp(f"{available + 1}-01-01", tz="UTC")
        target = work[(work[time_col] >= start) & (work[time_col] < end)].copy()
        year = available

    target["Samplingpoint"] = target["Samplingpoint"].astype("category")
    target["pollutant_median"] = np.float32(np.nan)
    categories = list(target["Samplingpoint"].cat.categories)
    for station in categories:
        st_all = work[work["Samplingpoint"] == station]
        st_curr = target[target["Samplingpoint"] == station]
        if st_all.empty or st_curr.empty:
            continue
        medians = compute_station_baseline(
            st_all,
            st_curr,
            time_col=time_col,
            value_col=value_col,
            neighbor_n=neighbor_n,
        )
        target.loc[st_curr.index, "pollutant_median"] = medians.to_numpy(dtype="float32")

    dust_mask = target["dust_flag"].fillna(False).astype(bool)
    target["Dust_contribution"] = np.where(
        dust_mask,
        target[value_col] - target["pollutant_median"],
        0.0,
    ).astype("float32")
    target["corrected_pollutant"] = np.where(
        dust_mask,
        target["pollutant_median"],
        target[value_col],
    ).astype("float32")
    negative = target["Dust_contribution"] < 0
    print(f"total negative days {int(negative.sum())}, and proceed to correction ->")
    target["Dust_contribution"] = target["Dust_contribution"].clip(lower=0)
    print(f"Now negative days {int((target['Dust_contribution'] < 0).sum())}")
    gc.collect()
    return target


def deduction_parquet_path(
    persistent_dir: str,
    pollutant: str,
    dataset_name: str,
    temporal_flag: str,
    year: int,
    neighbor_n: int,
) -> str:
    """Same name the notebook loads and saves on Google Drive."""
    return os.path.join(
        persistent_dir,
        f"CAMS_dust_{pollutant}_deduction_{dataset_name}_{temporal_flag}_{year}_MA{neighbor_n}.parquet",
    )


# Clock behind the ``day`` column of each saved deduction table.
DAY_DEFINITION = {
    "hour": "UTC day",
    "day": "station local day",
}


def tag_resolution(df: pd.DataFrame, temporal_flag: str) -> pd.DataFrame:
    """Mark which EEA product a deduction table came from.

    ``hour`` means ``day`` is a UTC day. ``day`` means ``day`` is the station's
    local calendar date. The two labels are not the same 24-hour window.
    """
    if temporal_flag not in DAY_DEFINITION:
        raise ValueError("temporal_flag must be 'hour' or 'day'.")
    out = df.copy()
    out["eea_resolution"] = temporal_flag
    out["day_definition"] = DAY_DEFINITION[temporal_flag]
    return out


def combine_hourly_and_daily(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    label_stations: bool = True,
) -> pd.DataFrame:
    """Stack the hourly and daily deduction tables for plotting.

    Rows are concatenated. They are not joined on ``day``, because that column
    is a UTC day in ``hourly`` and a station-local day in ``daily``. With
    ``label_stations``, ``Samplingpoint`` gains the clock in parentheses so a
    map or time series keeps the two products apart. The original id stays in
    ``Samplingpoint_id``.
    """
    if hourly is None or hourly.empty:
        raise ValueError("The hourly deduction table is empty.")
    if daily is None or daily.empty:
        raise ValueError("The daily deduction table is empty.")
    frames = [
        tag_resolution(hourly, "hour"),
        tag_resolution(daily, "day"),
    ]
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["Samplingpoint_id"] = out["Samplingpoint"].astype(str)
    if label_stations:
        out["Samplingpoint"] = out["Samplingpoint_id"] + " (" + out["day_definition"] + ")"
    return out


def load_and_combine_deduction_parquets(hourly_path: str, daily_path: str) -> pd.DataFrame:
    """Load the two saved deduction Parquet files and stack them."""
    missing = [path for path in (hourly_path, daily_path) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Both the hourly and the daily deduction Parquet files are required "
            "before they can be plotted together. Missing: " + ", ".join(missing)
        )
    hourly = pd.read_parquet(hourly_path)
    daily = pd.read_parquet(daily_path)
    combined = combine_hourly_and_daily(hourly, daily)
    print(
        f"Combined {len(hourly):,} hourly rows and {len(daily):,} daily rows "
        f"into {len(combined):,} rows. "
        "The hourly `day` is a UTC day; the daily `day` is the station local day."
    )
    return combined


def station_summary_table(
    df: pd.DataFrame,
    pollutant_threshold: float,
    samplingpoint_col: str = "Samplingpoint",
) -> pd.DataFrame:
    """Per-station summary used by the notebook maps.

    ``Dust_exceedance`` is a day when the measurement is above the limit and
    the corrected concentration is at or below it. Grouping follows
    ``samplingpoint_col``, so a combined table with clock labels in
    ``Samplingpoint`` keeps the hourly and daily products as separate rows.
    """
    work = df.copy()
    work["Dust_exceedance"] = (work["daily_mean"] > pollutant_threshold) & (
        work["corrected_pollutant"] <= pollutant_threshold
    )
    named = {
        "Latitude": ("Latitude", "first"),
        "Longitude": ("Longitude", "first"),
        "Altitude": ("Altitude", "first"),
        "Exceedance_days": ("Exceedance", "sum"),
        "Dust_exceedance_days": ("Dust_exceedance", "sum"),
        "Average_pollutant": ("daily_mean", "mean"),
        "Average_pollutant_dust_removed": ("corrected_pollutant", "mean"),
    }
    if "eea_resolution" in work.columns and samplingpoint_col != "eea_resolution":
        named["eea_resolution"] = ("eea_resolution", "first")
        named["day_definition"] = ("day_definition", "first")
    table = work.groupby(samplingpoint_col, as_index=False, observed=False).agg(**named)
    table["NonDust_exceedance"] = table["Exceedance_days"] - table["Dust_exceedance_days"]
    column_order = [
        samplingpoint_col,
        "eea_resolution",
        "day_definition",
        "Latitude",
        "Longitude",
        "Altitude",
        "Exceedance_days",
        "Dust_exceedance_days",
        "NonDust_exceedance",
        "Average_pollutant",
        "Average_pollutant_dust_removed",
    ]
    column_order = [col for col in column_order if col in table.columns]
    return table[column_order]


def metadata_path_for(persistent_dir: str, project_dir: str) -> str:
    candidates = [
        os.path.join(persistent_dir, "DataExtract.csv"),
        os.path.join(project_dir, "DataExtract.csv"),
        os.path.join(project_dir, "Data", "DataExtract.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def main() -> pd.DataFrame:
    """Run the daily deduction end to end. Used by ``python cams_data_to_eea_daily.py``."""
    start_time = time.time()
    try:
        import psutil

        print(
            f"{os.cpu_count()} Physical CPUs | {psutil.cpu_count(logical=True)} "
            "Logical Threads detected."
        )
    except Exception:
        print(f"{os.cpu_count()} CPUs detected.")

    flags = resolve_run_flags(
        download_eea=DOWNLOAD_EEA,
        download_cams=DOWNLOAD_CAMS,
        compute_cams_daily=COMPUTE_CAMS_DAILY,
        load_parquet=False,
        eea_temporal_flag=EEA_temporal_flag,
    )
    for note in flags["notes"]:
        print(note)
    if not flags["local_day"]:
        print(
            "EEA_temporal_flag is 'hour'. This script still averages CAMS on the "
            "station local day; the notebook UTC-grid path is the hourly option."
        )

    persistent_dir, _drive_on = resolve_output_dirs(project_dir, USE_GOOGLE_DRIVE, GDRIVE_OUTPUT_DIR)
    extract_dir = download_or_extract_eea(
        project_dir,
        persistent_dir,
        Countries,
        POLLUTANT,
        dataset,
        YEAR,
        EEA_temporal_flag,
        flags["download_eea"],
    )
    if flags["download_cams"] or flags["compute_cams_daily"]:
        download_and_extract_cams(
            project_dir,
            persistent_dir,
            YEAR,
            VAR,
            download=flags["download_cams"],
            extract=flags["compute_cams_daily"],
        )

    daily_eea = load_and_filter_eea_daily(extract_dir, EEA_temporal_flag)
    df_processed = filter_daily_by_coverage(
        daily_eea,
        reference_year=YEAR,
        min_pct=Station_Temporal_coverage,
        keep_coverage_columns=False,
    )
    print(f"Total stations valid after filtering: {df_processed['Samplingpoint'].nunique()}")
    meta = metadata_path_for(persistent_dir, project_dir)
    if not os.path.isfile(meta):
        raise FileNotFoundError(
            f"Metadata file DataExtract.csv not found at {meta}. "
            "Place it in the output folder, as the notebook does."
        )
    df_processed = merge_station_metadata(df_processed, meta, POLLUTANT)

    nc_files = list_cams_netcdf(project_dir, persistent_dir, VAR, YEAR)
    print(f"CAMS files: {nc_files}")
    df = add_cams_local_daily_dust(
        nc_files,
        df_processed,
        var_name=VAR,
        buffer=CAMS_BUFFER_DEG,
        cache_dir=persistent_dir,
        reuse_cache=True,
        recompute_daily=bool(flags["compute_cams_daily"]),
        min_hours=MIN_CAMS_HOURS,
        time_col="day",
        year=YEAR,
    )
    neighbor_MA = int (Basline_MA_days / 2)
    result = run_deduction_for_pollutant(
        df,
        year=YEAR,
        pollutant=POLLUTANT,
        dust_threshold=cams_dust_threshold,
        neighbor_n= neighbor_MA,
    )
    output = deduction_parquet_path(
        persistent_dir, POLLUTANT, dataset, EEA_temporal_flag, YEAR, Basline_MA_days
    )
    result.to_parquet(output, index=False, compression="snappy")
    elapsed = time.time() - start_time
    print("-" * 30)
    print(f"Total Duration: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    print(f"Rows processed: {len(result)}")
    print(f"save to {output}")
    print("-" * 30)
    return result


if __name__ == "__main__":
    main()
