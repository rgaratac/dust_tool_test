"""Processing steps of the simplified Dust deduction notebook.

The notebook ``Dust_deduction_tool_simple.ipynb`` only holds the options; every
step is a method of :class:`DustTool`, which keeps the intermediate tables
(``df_processed``, ``df``, ``pairing_session``, ``df_target_year``,
``station_table``) between cells. The steps are the same as in
``Dust_deduction_tool_full copy.ipynb``.
"""
from __future__ import annotations

import calendar
import gc
import glob
import importlib.util
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import cams_data_to_eea_daily as cams_daily
import natural_contribution as nc
import pairing as pr
import pairing_viz as pv
import util

# util.filter_daily_by_coverage uses calendar without importing it.
util.calendar = calendar

try:
    from IPython.display import display
except ImportError:  # plain Python: print instead
    display = print

ALL_COUNTRIES = ["AD", "AL", "AT", "BA", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GB",
                 "GR", "HR", "HU", "IE", "IS", "IT", "LT", "LU", "LV", "ME", "MK", "MT", "NL", "NO", "PL", "PT",
                 "RO", "RS", "SE", "SI", "SK", "XK"]


def in_colab() -> bool:
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


@dataclass
class Settings:
    """All user options of the notebook (sections 2 and 3)."""

    # Selection
    year: int = 2025
    pollutant: str = "PM10"                 # 'PM10' or 'PM2.5'
    dataset: str = "E1a"                    # 'E1a' (verified) or 'E2a' (up-to-date)
    temporal_flag: str = "day"              # 'hour' or 'day'
    countries: list = field(default_factory=lambda: ["ES"])
    # Run switches
    download_eea: bool = False
    download_cams: bool = False
    compute_cams_daily: bool = False
    load_parquet: bool = True
    project_dir: str = "."
    use_google_drive: bool | None = None    # None: on in Colab, off locally
    var: str = "dust"
    # Dust flag, limit values and coverage
    cams_dust_threshold: float = 5
    pm10_daily_limit: float = 45
    pm25_daily_limit: float = 25
    pm10_annual_limit: float = 20
    pm25_annual_limit: float = 10
    station_coverage: float = 65
    # Station pairing
    pairing_mode: str = "default"           # 'default' or 'none'
    max_rb_distance_km: float = 100
    max_rb_altitude_diff_m: float = 500
    max_ub_distance_km: float = 50
    pair_same_country: bool = True
    use_user_pairing: bool = False
    user_pairing_file: str = "station_pairing_user.csv"
    reapply_pairing_on_load: bool = True
    # Natural-contribution methods
    option1_half_window: int = 15
    option2_half_window: int = 3
    reference_percentile: float = 50
    min_reference_days_option1: int = 10
    min_reference_days_option2: int = 3


def eea_to_utc(series: pd.Series) -> pd.Series:
    """EEA hourly stamps are fixed UTC+1 (no daylight-saving time)."""
    s = pd.to_datetime(series, errors="coerce")

    def _localize_if_naive(x):
        if pd.isna(x):
            return x
        return x.tz_localize("Etc/GMT-1") if x.tzinfo is None else x

    return s.apply(_localize_if_naive).dt.tz_convert("UTC")


def load_eea_hourly_as_daily(folder: str) -> pd.DataFrame:
    """Valid, verified hourly EEA values to UTC daily means; only days with 24 hours are kept."""
    parquet_files = glob.glob(os.path.join(folder, "**/*.parquet"), recursive=True)
    if not parquet_files:
        raise FileNotFoundError(f"No EEA parquet files under {folder}. Set download_eea=True first.")
    needed_cols = ["Value", "Validity", "Verification", "AggType", "Start", "Samplingpoint", "Unit", "Pollutant"]
    frames = []
    for file in parquet_files:
        tmp = pd.read_parquet(file, columns=needed_cols)
        tmp["Value"] = pd.to_numeric(tmp["Value"], errors="coerce").astype("float32")
        tmp["Samplingpoint"] = tmp["Samplingpoint"].astype("category")
        tmp["Pollutant"] = tmp["Pollutant"].astype("category")
        frames.append(tmp)
    obs = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    obs["Validity"] = pd.to_numeric(obs["Validity"], errors="coerce").astype("int8")
    obs["Verification"] = pd.to_numeric(obs["Verification"], errors="coerce").astype("int8")
    obs["AggType"] = obs["AggType"].astype("category")
    obs["Start"] = eea_to_utc(obs["Start"])
    print(f"Observations before filtering: {obs.shape[0]:,}")

    mask = ((obs["Validity"] == 1) & (obs["Verification"] < 3)
            & (obs["AggType"].str.lower() == "hour") & (obs["Value"] >= 0))
    obs = obs.loc[mask].drop(columns=["Verification", "AggType"], errors="ignore")
    obs["day"] = obs["Start"].dt.floor("D")
    daily = (
        obs.groupby(["Pollutant", "day", "Samplingpoint"], sort=True, observed=False)
        .agg(daily_mean=("Value", "mean"), n_hours=("Value", "count"), unit=("Unit", "first"))
        .reset_index()
    )
    daily = daily[daily["n_hours"] == 24]
    del obs
    gc.collect()
    return daily


def cams_utc_daily_grid(project_dir: str, output_dir: str, year: int, var: str, compute: bool):
    """UTC daily mean CAMS grid for hourly EEA data: built from the hourly NetCDF or loaded from file."""
    import xarray as xr

    daily_path = os.path.join(output_dir, f"IRA_{var}", f"cams_dust_{year}_daily_mean.nc")
    if not compute:
        if not os.path.exists(daily_path):
            print(f"Daily mean file {daily_path} not found and compute_cams_daily is False.")
            return None
        print(f"Loading existing daily mean file from: {daily_path}")
        return xr.open_mfdataset(daily_path, chunks={"time": "auto"})

    netcdf_files = np.sort(glob.glob(os.path.join(project_dir, f"IRA_{var}", f"cams.eaq.ira.ENSa.dust*{year}*.nc")))
    print(f"Found {len(netcdf_files)} source files for processing.")
    datasets = []
    for f in netcdf_files:
        try:
            datasets.append(xr.open_dataset(f, chunks={"time": 24, "lat": 100, "lon": 100}))
        except Exception as exc:
            print(f"Error opening {f}: {exc}")
    if not datasets:
        print("No source datasets were found to process.")
        return None
    master_lat, master_lon = datasets[0].lat, datasets[0].lon
    aligned = [ds if (ds.lat.equals(master_lat) and ds.lon.equals(master_lon))
               else ds.interp(lat=master_lat, lon=master_lon, method="nearest") for ds in datasets]
    cams = xr.concat(aligned, dim="time").sortby("time")
    print("Resampling to daily mean...")
    daily = cams.resample(time="1D").mean(keep_attrs=False).compute()
    # Attributes are dropped: some contain characters that break to_netcdf.
    daily.attrs = {}
    for name in daily.variables:
        daily[name].attrs = {}
    os.makedirs(os.path.dirname(daily_path), exist_ok=True)
    daily.to_netcdf(daily_path)
    print(f"Saved daily means to {daily_path}")
    cams.close()
    return daily


def setup_cds_key() -> None:
    """Ask once for the Copernicus ADS API key and store it in ~/.cdsapirc."""
    path = os.path.expanduser("~/.cdsapirc")
    if os.path.exists(path):
        print(f"Skipping: configuration file already exists at {path}")
        return
    from getpass import getpass

    api_key = getpass("Enter your Copernicus ADS API Key: ").strip()
    with open(path, "w") as f:
        f.write(f"url: https://ads.atmosphere.copernicus.eu/api\nkey: {api_key}\n")
    print(f"Success! .cdsapirc has been created at {path}")


class DustTool:
    """Holds the settings and the tables of one run; one method per notebook step."""

    def __init__(self, settings: Settings):
        s = self.settings = settings
        if s.pairing_mode not in ("default", "none"):
            raise ValueError("pairing_mode must be 'default' or 'none'")
        self.in_colab = in_colab()
        use_drive = self.in_colab if s.use_google_drive is None else s.use_google_drive
        self.output_dir, self.use_google_drive = cams_daily.resolve_output_dirs(s.project_dir, use_drive and self.in_colab)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output folder: {os.path.abspath(self.output_dir)}")

        # DOWNLOAD_CAMS brings new hourly files, so the daily product has to be rebuilt.
        run = cams_daily.resolve_run_flags(
            download_eea=s.download_eea, download_cams=s.download_cams,
            compute_cams_daily=s.compute_cams_daily, load_parquet=s.load_parquet,
            eea_temporal_flag=s.temporal_flag,
        )
        s.download_eea, s.download_cams = run["download_eea"], run["download_cams"]
        s.compute_cams_daily, s.load_parquet = run["compute_cams_daily"], run["load_parquet"]
        for note in run["notes"]:
            print(note)
        print("EEA daily data: CAMS daily means use each station's local day." if run["local_day"]
              else "EEA hourly data: CAMS daily means use UTC days.")

        self.limits = nc.LimitValues(s.pm10_daily_limit, s.pm25_daily_limit, s.pm10_annual_limit, s.pm25_annual_limit)
        self.daily_limit = self.limits.daily(s.pollutant)
        self.annual_limit = self.limits.annual(s.pollutant)
        self.pairing_limits = pr.PairingLimits(
            max_rb_distance_km=s.max_rb_distance_km, max_rb_altitude_diff_m=s.max_rb_altitude_diff_m,
            max_ub_distance_km=s.max_ub_distance_km, same_country=s.pair_same_country,
        )
        self.params = nc.MethodParams(
            option1_half_window=s.option1_half_window, option2_half_window=s.option2_half_window,
            percentile=s.reference_percentile, min_reference_days_option1=s.min_reference_days_option1,
            min_reference_days_option2=s.min_reference_days_option2,
        )
        print(f"{s.pollutant}: daily limit {self.daily_limit} µg/m3, annual limit {self.annual_limit} µg/m3")
        print(f"Pairing mode: {s.pairing_mode}")

        self.dataset = s.dataset  # replaced by the user label with own observations
        self.eea_folder = cams_daily.eea_extract_dir(s.project_dir, s.pollutant, s.year, s.dataset, s.temporal_flag)
        self.user_station_metadata = None
        self.use_user_observations = False
        self.daily_obs = None
        self.df_processed = None
        self.daily_cams = None
        self.df = None
        self.pairing_session = None
        self.station_pairing = None
        self.pairing_metadata = None
        self.df_target_year = None
        self.station_table = None

    # ---------- section 3: data ----------
    def download_eea(self) -> None:
        s = self.settings
        self.eea_folder = cams_daily.download_or_extract_eea(
            s.project_dir, self.output_dir, s.countries, s.pollutant, s.dataset, s.year,
            s.temporal_flag, download=s.download_eea,
        )

    def load_user_observations(self, enabled=False, observations_file="user_observations.csv",
                               stations_file="user_stations.csv", upload=False, label="user") -> None:
        """Use your own observations (section 3.1.1) instead of the EEA download."""
        self.use_user_observations = enabled
        self.user_station_metadata = None
        if not enabled:
            print("Using the EEA observations.")
            return
        s = self.settings
        obs_path = os.path.join(self.output_dir, observations_file)
        stations_path = os.path.join(self.output_dir, stations_file)
        if upload:
            print("Choose the observations file:")
            pv.upload_file(obs_path)
            print("Choose the stations file:")
            pv.upload_file(stations_path)
        missing = [p for p in (obs_path, stations_path) if not os.path.exists(p)]
        if missing:
            examples = cams_daily.write_user_templates(self.output_dir, s.temporal_flag, s.pollutant)
            raise FileNotFoundError(
                "Not found: " + ", ".join(missing) + "\nExample files were written to:\n  " + "\n  ".join(examples)
                + f"\nFill them in, save them as {observations_file} and {stations_file} and run this cell again."
            )
        self.user_station_metadata = cams_daily.load_user_stations(stations_path, s.pollutant)
        print(f"{len(self.user_station_metadata)} stations read from {stations_path}")
        if not s.load_parquet:
            self.daily_obs = cams_daily.load_user_observations(
                obs_path, self.user_station_metadata, temporal_flag=s.temporal_flag, pollutant=s.pollutant,
            )
            display(self.daily_obs.head())
        self.dataset = label
        print(f"Using your own observations; output files are labelled {label!r}.")

    def download_cams(self) -> None:
        s = self.settings
        if s.download_cams and importlib.util.find_spec("cdsapi") is None:
            raise ImportError('Install cdsapi first: %pip install "cdsapi>=0.7.7"')
        if s.download_cams or s.compute_cams_daily:
            # Zips stay in the output folder. Hourly NetCDF is extracted locally for this run only.
            cams_daily.download_and_extract_cams(
                s.project_dir, self.output_dir, s.year, s.var,
                download=s.download_cams, extract=s.compute_cams_daily,
            )
        else:
            print("compute_cams_daily is False. Zips are not extracted; the saved daily CAMS file is used later.")

    # ---------- section 4: data preparation ----------
    def _metadata_path(self):
        candidates = [os.path.join(self.output_dir, "DataExtract.csv"),
                      os.path.join(self.settings.project_dir, "DataExtract.csv"),
                      os.path.join(self.settings.project_dir, "Data", "DataExtract.csv")]
        return next((p for p in candidates if os.path.exists(p)), None), candidates

    def prepare_observations(self) -> None:
        """Validity filter, daily means, coverage filter and station coordinates (sections 4.1-4.3)."""
        s = self.settings
        if s.load_parquet:
            print("load_parquet is True: the saved results are loaded in section 7.")
            return
        if not self.use_user_observations:
            if s.temporal_flag == "hour":
                self.daily_obs = load_eea_hourly_as_daily(self.eea_folder)
            else:
                # Daily EEA stamps are station local time; they are not converted to UTC.
                self.daily_obs = cams_daily.load_and_filter_eea_daily(self.eea_folder, temporal_flag="day")
            print(f"Stations in the EEA data: {self.daily_obs['Samplingpoint'].nunique()}")

        df = util.filter_daily_by_coverage(self.daily_obs, reference_year=s.year,
                                           min_pct=s.station_coverage, keep_coverage_columns=False)
        df["Exceedance"] = df["daily_mean"] > self.daily_limit
        print(f"Stations with at least {s.station_coverage}% coverage in {s.year}: {df['Samplingpoint'].nunique()}")

        metadata_path, candidates = self._metadata_path()
        if metadata_path or self.user_station_metadata is not None:
            # Stations from the user stations file come first and replace DataExtract.csv rows.
            metadata = cams_daily.combine_station_metadata(metadata_path, self.user_station_metadata)
            metadata["Samplingpoint"] = metadata["Air Quality Station EoI Code"].str[:2] + "/" + metadata["Sampling Point Id"]
            geo = metadata[["Samplingpoint", "Longitude", "Latitude", "Altitude", "Timezone"]].drop_duplicates("Samplingpoint")
            df = df.merge(geo, on="Samplingpoint", how="left")
            print(f"Metadata merged from: {metadata_path or 'the user stations file'}")
            if self.use_user_observations:
                no_coords = df.loc[df["Latitude"].isna(), "Samplingpoint"].unique()
                if len(no_coords):
                    print(f"{len(no_coords)} station(s) without coordinates are dropped: {list(no_coords)[:10]}")
                    df = df.dropna(subset=["Latitude", "Longitude"])
        else:
            print(f"Warning: DataExtract.csv not found in {candidates}. Skipping spatial merge.")
        self.df_processed = df

    def match_cams(self) -> None:
        """Daily CAMS dust at each station (sections 4.4-4.5): UTC days for hourly data, local days for daily data."""
        s = self.settings
        if s.load_parquet:
            return
        if s.temporal_flag == "hour":
            self.daily_cams = cams_utc_daily_grid(s.project_dir, self.output_dir, s.year, s.var, s.compute_cams_daily)
            if self.daily_cams is None:
                raise FileNotFoundError("No daily CAMS grid: run with compute_cams_daily=True.")
            lat, lon = self.daily_cams.lat, self.daily_cams.lon
            df = self.df_processed
            before = df["Samplingpoint"].nunique()
            df = df[df["Latitude"].between(lat.min().item(), lat.max().item())
                    & df["Longitude"].between(lon.min().item(), lon.max().item())].copy()
            print(f"{before - df['Samplingpoint'].nunique()} stations outside the CAMS domain removed, "
                  f"{df['Samplingpoint'].nunique()} remain.")
            self.df = util.add_cams_daily_dust_by_station(
                cams_ds=self.daily_cams, df=df, var_name="dust", time_col="day",
                lat_col="Latitude", lon_col="Longitude", station_col="Samplingpoint", spatial_method="linear",
            )
        else:
            nc_files = cams_daily.list_cams_netcdf(s.project_dir, self.output_dir, s.var, s.year)
            print(f"CAMS files for local-day means: {len(nc_files)}")
            self.df = cams_daily.add_cams_local_daily_dust(
                nc_files, self.df_processed, var_name=s.var, buffer=0.2, cache_dir=self.output_dir,
                reuse_cache=True, recompute_daily=bool(s.compute_cams_daily), min_hours=18,
                time_col="day", year=s.year,
            )
        display(self.df.head())

    # ---------- section 4.6: pairing ----------
    @property
    def pairing_user_path(self) -> str:
        return os.path.join(self.output_dir, self.settings.user_pairing_file)

    @property
    def deduction_parquet(self) -> str:
        s = self.settings
        # When loading, a file saved by an earlier version (name ending in _MA6) is used if the new name is missing.
        find = cams_daily.find_deduction_parquet if s.load_parquet else cams_daily.deduction_parquet_path
        return find(self.output_dir, s.pollutant, self.dataset, s.temporal_flag, s.year)

    def build_pairing(self) -> None:
        s = self.settings
        metadata_path, candidates = self._metadata_path()
        if metadata_path is None and self.user_station_metadata is None:
            raise FileNotFoundError(f"DataExtract.csv not found in {candidates}")

        # Only stations with data can be references: this run's stations, or those in the saved Parquet.
        if not s.load_parquet:
            station_ids = self.df["Samplingpoint"].astype(str).unique()
        elif os.path.exists(self.deduction_parquet):
            station_ids = pd.read_parquet(self.deduction_parquet, columns=["Samplingpoint"])["Samplingpoint"].astype(str).unique()
        else:
            station_ids = None
            print(f"{self.deduction_parquet} not found; pairing all stations of the selected countries.")

        self.pairing_metadata = cams_daily.combine_station_metadata(metadata_path, self.user_station_metadata)
        stations = pr.build_station_table(self.pairing_metadata, pollutant=s.pollutant,
                                          station_ids=station_ids, countries=s.countries)
        if s.pairing_mode == "none":
            # RB stations use Option 1 on their own data, all others Option 2 self-calibration.
            default = pr.no_pairing(stations)
            self.station_pairing = default.copy()
            print(f"{len(stations)} stations classified; no pairing "
                  f"(RB: Option 1 ±{s.option1_half_window} d, others: Option 2 ±{s.option2_half_window} d).")
        else:
            default = pr.default_pairing(stations, self.pairing_limits)
            default_path = os.path.join(self.output_dir, pr.DEFAULT_PAIRING_FILENAME)
            pr.export_pairing_csv(default, default_path)
            print(f"{len(stations)} stations classified; default pairing saved to {default_path}")
            n_unknown = int((stations["class"] == "unknown").sum())
            if n_unknown:
                print(f"{n_unknown} station(s) have an unknown area/type and get method none.")
            self.station_pairing, report = pr.load_pairing(
                stations, default, self.pairing_user_path,
                use_user_pairing=s.use_user_pairing, limits=self.pairing_limits,
            )
            if report is not None:
                print(f"Using {self.pairing_user_path}")
                print(report.summary())
                display(report.changes)
        # The deduction in section 6 reads pairing_session.table.
        self.pairing_session = pv.PairingSession(
            stations, default, self.station_pairing, limits=self.pairing_limits,
            autosave_path=self.pairing_user_path if s.pairing_mode == "default" else None,
        )
        display(pr.method_counts(self.pairing_session.table))
        display(self.pairing_session.table.head())

    def pairing_app(self, clickable_map=True):
        return pv.pairing_app(self.pairing_session, countries=self.settings.countries, clickable_map=clickable_map)

    def export_pairing_csv(self) -> None:
        self.pairing_session.export_csv(self.pairing_user_path)
        print(f"Pairing table ({len(self.pairing_session.table)} stations) written to {self.pairing_user_path}")
        pv.download_file(self.pairing_user_path)

    def load_pairing_csv(self, upload=False) -> None:
        if upload:
            pv.upload_pairing_csv(self.pairing_user_path)
        if not os.path.exists(self.pairing_user_path):
            print(f"No edited pairing found at {self.pairing_user_path}.")
            return
        try:
            report = self.pairing_session.load_csv(self.pairing_user_path)
        except pr.PairingValidationError as exc:
            print(exc)
            print("The pairing was not changed. Fix the CSV and run this cell again.")
            return
        print(report.summary())
        display(report.changes)
        display(pr.method_counts(self.pairing_session.table))

    # ---------- sections 5-6: dust days and deduction ----------
    def flag_dust_days(self) -> None:
        if self.settings.load_parquet:
            return
        self.df["dust_flag"] = self.df["cams_dust"] > self.settings.cams_dust_threshold
        self.df["Exceedance"] = self.df["daily_mean"] > self.daily_limit
        display(self.df.head())
        negative = self.df[self.df["daily_mean"] < 0]
        print(f"Rows with negative daily_mean: {len(negative)}")
        if not negative.empty:
            display(negative.sort_values("daily_mean").head(10))

    def run_deduction(self) -> None:
        s = self.settings
        if s.load_parquet:
            return
        self.df_target_year = nc.run_natural_deduction(
            self.df, self.pairing_session.table, s.year, pollutant=s.pollutant,
            dust_threshold=s.cams_dust_threshold, params=self.params, limits=self.limits,
        )
        gc.collect()
        display(self.df_target_year[nc.NEW_COLUMNS].head())

    def deduction_summary(self) -> None:
        """Event days per class and method, and the unconfirmed dust days (section 6.1)."""
        if self.settings.load_parquet:
            return
        d = self.df_target_year
        display(
            d[d["cams_dust_flag"]].groupby(["station_class", "method"], observed=True).agg(
                stations=("Samplingpoint", "nunique"),
                cams_dust_days=("cams_dust_flag", "sum"),
                confirmed_dust_days=("confirmed_dust_day", "sum"),
                insufficient_reference=("insufficient_reference", "sum"),
                mean_contribution=("natural_contribution", "mean"),
            )
        )
        cams_days = int(d["cams_dust_flag"].sum())
        confirmed = int(d["confirmed_dust_day"].sum())
        insufficient = int(d["insufficient_reference"].sum())
        print(f"CAMS dust days: {cams_days}, confirmed: {confirmed}, "
              f"not confirmed: {cams_days - confirmed - insufficient}, insufficient reference: {insufficient}")
        print(f"Negative contributions: {int((d['natural_contribution'] < 0).sum())}")

    # ---------- section 7: results ----------
    def save_or_load(self) -> None:
        s = self.settings
        path = self.deduction_parquet
        if not s.load_parquet:
            self.df_target_year.to_parquet(path, compression="snappy")
            print(f"Saved {path}")
        else:
            print(f"Open {path}")
            self.df_target_year = pd.read_parquet(path)
            if s.reapply_pairing_on_load:
                self.df_target_year = nc.compute_natural_contribution(
                    self.df_target_year, self.pairing_session.table, dust_threshold=s.cams_dust_threshold,
                    params=self.params, daily_limit=self.daily_limit,
                )
                print("Deduction recomputed with the current pairing.")

    def make_station_table(self, save_csv=True) -> pd.DataFrame:
        s = self.settings
        if "adjusted_value" in self.df_target_year.columns:
            self.station_table = nc.station_indicators(
                self.df_target_year, daily_limit=self.daily_limit, annual_limit=self.annual_limit)
        else:
            # Parquet from an earlier version, loaded with reapply_pairing_on_load = False.
            self.station_table = cams_daily.station_summary_table(self.df_target_year, self.daily_limit)
        display(self.station_table.head())
        if save_csv:
            key = "muti_countries" if len(s.countries) >= 2 else s.countries[0]
            path = os.path.join(self.output_dir, f"{key}_station_annual_{s.pollutant}_{self.dataset}_{s.temporal_flag}_{s.year}.csv")
            self.station_table.to_csv(path)
            print(f"Saved {path}")
        return self.station_table

    def threshold_sweep(self, thresholds, group="country", min_share=0.9, min_days=10):
        """Share of CAMS dust days with a contribution > 0 per threshold, and the suggested threshold."""
        import dust_threshold_sensitivity as sens

        sweep_input = self.df if not self.settings.load_parquet else self.df_target_year
        parts = []
        for threshold in thresholds:
            frame = nc.compute_natural_contribution(
                sweep_input, self.pairing_session.table, dust_threshold=threshold,
                params=self.params, daily_limit=self.daily_limit,
            )
            parts.append(sens.contribution_consistency({float(threshold): frame}, group_col=group))
            del frame
            gc.collect()
        consistency = pd.concat(parts, ignore_index=True)
        suggestion = sens.lowest_consistent_threshold(consistency, group_col=group,
                                                      min_share=min_share, min_days=min_days)
        display(consistency)
        display(suggestion)
        return consistency, suggestion

    def combine_hourly_and_daily(self) -> None:
        """Stack the saved hourly and daily Parquet files for the plots (section 7.3)."""
        s = self.settings

        def _path(flag):
            return cams_daily.find_deduction_parquet(self.output_dir, s.pollutant, self.dataset, flag, s.year)

        self.df_target_year = cams_daily.load_and_combine_deduction_parquets(_path("hour"), _path("day"))
        self.station_table = cams_daily.station_summary_table(self.df_target_year, self.daily_limit)
        print("The plots use both products. Samplingpoint labels include the clock "
              "(UTC day or station local day) so the two series stay separate.")
        display(self.df_target_year.groupby(["eea_resolution", "day_definition"], observed=False).size())
        display(self.station_table.head())
