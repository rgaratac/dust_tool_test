# CAMS Natural Dust Deduction Tool

This repository estimates how much of the particulate matter measured at European air-quality stations comes from natural Saharan and other desert dust, and subtracts that contribution from the measurements.

The main workflow is the notebook `Dust_deduction_tool_full.ipynb` (the CAMS Natural Dust Service Tool). It supports the natural-source assessment in Article 16 of the Ambient Air Quality Directive (Directive (EU) 2024/2881): identifying where limit-value exceedances are attributable to natural sources, and removing that natural dust from reported concentrations. The method follows the European Commission (DG ENV) guidance on assessing natural dust contributions ([sec 2011-0208](https://data.consilium.europa.eu/doc/document/ST%206771%202011%20INIT/EN/pdf)). Days affected by dust are identified with the CAMS European air-quality interim reanalysis dust product. Following the EU methodology for natural contributions, stations are classified as regional background (RB), urban background (UB) or traffic/industrial (TH), paired with nearby reference stations, and the natural load is quantified at the paired RB (Option 1) or by UB self-calibration (Option 2). Member States can inspect and edit the pairing on a map or as a CSV.

Created by Jessie Zhang, TNO.

## Notebook workflow

The notebook runs in six stages. The chart below is the hourly workflow, drawn from the current notebook. With `EEA_temporal_flag = 'day'` the UTC daily-grid step is replaced by a per-station local-day CAMS mean (section 4.5).

![Dust deduction tool notebook workflow](docs/Dust_deduction_tool_flowchart.png)

### 1. Setup and parameters

Section 1 installs `cartopy`, downloads any missing helper module (`util.py`, `cams_data_to_eea_daily.py`, `pairing.py`, `natural_contribution.py`, `pairing_viz.py`, `dust_threshold_sensitivity.py`) from `REPO_RAW_BASE`, and imports them.

Section 2 is where you set the run:

| Setting | Role |
|---|---|
| `Countries` | One or more EEA country codes (for example `["ES"]`, or the full list of reporting countries in the notebook) |
| `YEAR` | Calendar year to analyse |
| `POLLUTANT` | `PM10` or `PM2.5` |
| `dataset` | `E1a` (verified data, reported annually, available from 2013) or `E2a` (up-to-date unverified data, available from 2023) |
| `EEA_temporal_flag` | `hour` or `day`. This chooses the CAMS average. `hour` treats EEA hourly values as fixed UTC+1 and averages CAMS on UTC days. `day` uses EEA’s own daily values, which are in station local time, and averages CAMS over that local day |
| `USE_GOOGLE_DRIVE` | If `True` (typical in Colab), files are stored under `/content/drive/MyDrive/CAMS_Tool_output`. If `False`, they stay in the project folder |

Downloads and the heavy processing only need to run once. Later runs reuse the saved files:

| Flag | First run | Later runs | What it does |
|---|---|---|---|
| `DOWNLOAD_EEA` | `True` | `False` | Downloads EEA observations and stores them as a zip |
| `DOWNLOAD_CAMS` | `True` | `False` | Downloads any missing CAMS zip into the output folder. This also turns `COMPUTE_CAMS_DAILY` on |
| `COMPUTE_CAMS_DAILY` | `True` | `False` | Extracts the stored zips and rebuilds the daily CAMS product. `hour` writes a UTC daily grid. `day` rebuilds the per-station local-day table; monthly station caches are still reused |
| `LOAD_PARQUET_DATA` | `False` | `True` | Loads the saved daily-results Parquet file and skips recomputation. Turned off automatically when `DOWNLOAD_EEA`, `DOWNLOAD_CAMS`, or `COMPUTE_CAMS_DAILY` is True |

Thresholds, limit values and method parameters in the next cell:

| Parameter | Default | Meaning |
|---|---|---|
| `CAMS_dust_threshold` | 5 µg/m³ | A day is a natural-event (dust) day when CAMS dust at the station is above this value |
| `PM10_DAILY_LIMIT` | 45 µg/m³ | Daily PM10 limit (Directive (EU) 2024/2881). Set 50 for the previous limit |
| `PM25_DAILY_LIMIT` | 25 µg/m³ | Daily PM2.5 limit |
| `PM10_ANNUAL_LIMIT` | 20 µg/m³ | Annual PM10 limit |
| `PM25_ANNUAL_LIMIT` | 10 µg/m³ | Annual PM2.5 limit |
| `Station_Temporal_coverage` | 65% | Minimum share of days in the selected year required to keep a station |
| `Basline_MA_days` | 6 | Only used in output file names (`_MA6`) since the new methods |
| `MAX_RB_DISTANCE_KM` | 100 | Maximum horizontal distance to a suitable regional background (RB) reference |
| `MAX_RB_ALTITUDE_DIFF_M` | 500 | Maximum altitude difference to a suitable RB reference |
| `MAX_UB_DISTANCE_KM` | 50 | Maximum distance between a traffic/industrial (TH) station and its urban background (UB) reference |
| `PAIR_SAME_COUNTRY` | True | Only pair stations of the same country |
| `USE_USER_PAIRING` | False | Use `station_pairing_user.csv` from the output folder instead of the default pairing |
| `USER_PAIRING_FILE` | `station_pairing_user.csv` | Name of the edited pairing file. Use a different name per pollutant: sampling-point ids differ between PM10 and PM2.5 |
| `REAPPLY_PAIRING_ON_LOAD` | True | With `LOAD_PARQUET_DATA`, recompute the deduction on the loaded table with the current pairing |
| `OPTION1_HALF_WINDOW` | 15 | Option 1 reference window: d−15 … d+15 |
| `OPTION2_HALF_WINDOW` | 3 | Option 2 reference window: d−3 … d+3, or 3 days before/after a multi-day episode |
| `REFERENCE_PERCENTILE` | 50 | Percentile of the reference window |
| `MIN_REFERENCE_DAYS_OPTION1` | 10 | Minimum valid reference days for Option 1 |
| `MIN_REFERENCE_DAYS_OPTION2` | 3 | Minimum valid reference days for Option 2 |

`POLLUTANT_daily_threshold` and `POLLUTANT_annual_threshold` are set from these limits for the selected `POLLUTANT`.

### 2. Input data

Two datasets are downloaded, plus station metadata that you supply yourself.

**EEA observations.** When `DOWNLOAD_EEA` is `True`, the notebook requests Parquet data from the EEA air-quality download API for the selected countries, pollutant, dataset, and time resolution. The request runs from 14 December of the previous year through 31 December of the selected year, so the background window can be computed for the first days of the year. The zip is named `{dataset}_{POLLUTANT}_{EEA_temporal_flag}_{YEAR}.zip` and extracted under `EEA_{POLLUTANT}/{YEAR}/{dataset}/{EEA_temporal_flag}`. Large requests (many countries, or hourly data) can take several minutes. If `DOWNLOAD_EEA` is `False`, an existing zip is extracted instead.

**CAMS dust.** Downloading requires a free Copernicus Atmosphere Data Store account, an API key written to `~/.cdsapirc`, and acceptance of the licence for `cams-europe-air-quality-reanalyses`. The notebook asks for the key on the first download. It retrieves the ensemble interim-reanalysis surface dust field for December of the previous year and for each quarter of the selected year (five zip files). Those zips are the stored copy: `IRA_dust/CAMS_IRA_*.zip` on Google Drive when `USE_GOOGLE_DRIVE` is True, otherwise in the project folder. A zip that exists only next to a local extract is copied into that folder. Hourly NetCDF is extracted onto local disk only while `COMPUTE_CAMS_DAILY` is True, and is not written back to Drive. An existing zip is reused; only a missing zip is downloaded. `DOWNLOAD_CAMS = True` turns `COMPUTE_CAMS_DAILY` on, because a new zip has to be averaged. The average follows `EEA_temporal_flag`.

**Station metadata.** EEA observations identify a station only by sampling-point id. Latitude, longitude, altitude, and time zone come from a CSV downloaded manually from the [EEA Air Quality Viewer](https://discomap.eea.europa.eu/App/AQViewer/index.html?fqn=Airquality_Dissem.b2g.measurements). Name the file `DataExtract.csv` and place it in the output folder (`CAMS_Tool_output` on Google Drive, or the project folder when `USE_GOOGLE_DRIVE` is `False`). A 2024 extract is included in this repository at `Data/DataExtract.csv`; copy or point the notebook at that file if you want to reuse it. The notebook builds the join key as the first two characters of `Air Quality Station EoI Code`, a slash, and `Sampling Point Id`.

### 3. Data preparation

For the hourly path (`EEA_temporal_flag == 'hour'`), and only when results are not being loaded from Parquet:

1. Keep measurements with validity 1, verification 1 or 2, hourly aggregation, and a non-negative value. EEA hourly timestamps are treated as CET (fixed UTC+1) and converted to UTC.
2. Average each station-day, and keep only days that have all 24 hours.
3. Keep stations whose coverage in `YEAR` is at least `Station_Temporal_coverage`. Rows outside that year are kept for those stations so the background window still has the days before 1 January.
4. Merge station coordinates from `DataExtract.csv`.
5. If `COMPUTE_CAMS_DAILY` is `True`, combine the hourly CAMS NetCDF files and average them to UTC daily means, saved as `IRA_dust/cams_dust_{YEAR}_daily_mean.nc`. Otherwise the saved daily file is loaded. This step can exceed Colab memory; the notebook tells you to run it once on a machine with more memory and copy the NetCDF back.
6. Drop stations outside the CAMS domain, then linearly interpolate daily CAMS dust to each station (`add_cams_daily_dust_by_station` in `util.py`).

For the daily path (`EEA_temporal_flag = 'day'`):

1. Load the same EEA Parquet extract. Keep validity 1, verification below 3, aggregation `day`, and non-negative values. `Start` is left as the station’s local calendar date. It is not passed through `eea_to_utc`.
2. Apply the same coverage filter and merge `DataExtract.csv`, including `Timezone`.
3. Skip the full-domain UTC daily grid. `add_cams_local_daily_dust` in `cams_data_to_eea_daily.py` reads hourly CAMS one UTC day at a time, interpolates only the station locations, and averages each station over its own local day (see [Time zones](#time-zones) below).
4. The table that comes back has the same columns the hourly path uses (`day`, `daily_mean`, `cams_dust`, coordinates). Dust flags, the background median, the corrected concentration, the station summary, and the plots then run unchanged.

### Time zones

EEA hourly stamps are naive clock times that the notebook treats as fixed UTC+1 (`Etc/GMT-1`, no daylight-saving time) and converts to UTC with `eea_to_utc`. The daily mean is then a UTC day. That conversion is the wrong one for EEA daily values: those stamps are already the local time of the station, and a blanket UTC+1 shift would move a `UTC+02` or `UTC` station onto another date.

`Timezone` in `DataExtract.csv` is a fixed offset such as `UTC`, `UTC+01`, or `UTC-04`. For a station with offset *h* hours, the CAMS mean for local day D is the mean of the UTC hours in `[D 00:00 − h, D+1 00:00 − h)`. The join to the observation is that local calendar date. A day with fewer than 18 equivalent hours of CAMS data (75% of a 24-hour day; the same fraction is used if the file is not hourly) is left missing rather than filled from the nearest day. Daylight-saving transitions are not applied, which matches the hourly path’s fixed offset.

### Memory and Drive cache

The daily path never loads a full hourly CAMS grid. Each NetCDF file is opened lazily, cut to the station bounding box and the dust variable, and read one UTC day at a time as float32. Only the interpolated station series is kept. After each day the day-sized array is released.

Intermediate files follow the notebook’s Drive layout (`/content/drive/MyDrive/CAMS_Tool_output` when `USE_GOOGLE_DRIVE` is `True`, otherwise the project folder):

| File | Role |
|---|---|
| `IRA_dust/CAMS_IRA_*.zip` | The stored CAMS download (Drive when `USE_GOOGLE_DRIVE` is True). Kept between runs. Extracted only when `COMPUTE_CAMS_DAILY` is True |
| `IRA_dust/cams_dust_station_hourly_<stations>_<source>.parquet` | Hourly dust at the stations for one CAMS file. Reused on the next run, including when `COMPUTE_CAMS_DAILY` is `True` |
| `IRA_dust/cams_dust_local_daily_stations_<stations>_<year>_h18.parquet` | Local-day means. Reused when `COMPUTE_CAMS_DAILY` is `False` |

`<stations>` is a short hash of the station coordinates and time-zone offsets, so a different country list does not reuse the wrong cache. December of the previous year is included, because a positive offset reaches back into 31 December and the EEA request itself starts on 14 December.

### 4. Station classification and pairing (notebook section 4.6)

Stations are classified from `Air Quality Station Type` and `Air Quality Station Area` in `DataExtract.csv`:

| Class | Rule |
|---|---|
| RB, regional background | rural* area (`rural`, `rural-near_city`, `rural-regional`, `rural-remote`) and background type |
| UB, urban background | urban or suburban area and background type |
| TH, hotspot | traffic or industrial type |

The default pairing (`pairing.default_pairing`) gives every non-RB station the nearest RB station of the same country within `MAX_RB_DISTANCE_KM` and `MAX_RB_ALTITUDE_DIFF_M`, and every TH station the nearest UB station within `MAX_UB_DISTANCE_KM`. An RB station is its own reference. Only stations with data in the run are candidates. The method follows from the pairing:

| Method | When |
|---|---|
| `option1` | a suitable RB exists |
| `option2` | UB station without suitable RB |
| `option2_ub_paired` | TH station without suitable RB, with a paired UB |
| `none` | no valid reference; nothing is subtracted |

The pairing table has the columns `station_id, station_name, country, class, area, type, latitude, longitude, altitude, paired_rb_id, paired_rb_name, rb_distance_km, rb_altitude_diff_m, paired_ub_id, paired_ub_name, ub_distance_km, method, source, comment`. It is saved as `station_pairing_default.csv` in the output folder.

**Map.** `pairing_viz.plot_station_pairing` (Plotly) and `pairing_viz.plot_station_pairing_static` (Cartopy) show every station coloured by class, a green line to its RB and a thinner orange (dashed in the static map) line from a TH station to its UB. Stations without a usable reference are grey. Hover text gives the id and name, class, area, altitude, paired stations, distance, altitude difference, method and whether the pairing is default or manual. The interactive version has country and class filters. `util.plot_station_pairing` wraps both.

**Editing in Colab.** `pairing_viz.pairing_app` combines the map with a form: pick a station (click it or search by id or name), choose its RB and/or UB from candidates sorted by distance (with distance and altitude difference), or *none / force Option 2*, and press **Save**. The station is marked `source = manual`, distances and method are recomputed, the map redraws and the table is written to `station_pairing_user.csv`. **Reset station** and **Reset all** go back to the default.

**Editing as CSV.** Export the table (downloaded automatically in Colab), edit `paired_rb_id`, `paired_ub_id`, `method` or `comment`, and upload it again or place it in the output folder as `station_pairing_user.csv`. On loading (`pairing.apply_user_pairing`), every id must exist and a paired RB/UB must really be an RB/UB, otherwise the file is rejected with a list of errors. Pairs beyond the distance or altitude limits only produce warnings. Names, distances and altitude differences are recomputed, changed stations are marked `manual`, and the changes against the default pairing are listed.

The deduction always reads the current pairing table (`pairing_session.table`), so an edit made with the form or the CSV is used the next time section 6 runs, or when results are loaded with `REAPPLY_PAIRING_ON_LOAD`.

### 5. Natural-contribution deduction (notebook section 6)

Event days are the CAMS dust days (`cams_dust_flag`: CAMS dust above `CAMS_dust_threshold`). They are excluded from every reference window, not only the target day. The calculation (`natural_contribution.py`) uses the whole download period from 14 December of the previous year, then keeps the selected year.

* **Option 1 (regional background, preferred).** At the paired RB station, for each event day *d*: window *d*−15 … *d*+15, event days removed, 50th percentile. Net natural load = max(RB(*d*) − reference, 0). The same daily load is subtracted at every station paired to that RB, including the RB itself.
* **Option 2 (self-calibration).** For UB stations without a suitable RB: window *d*−3 … *d*+3 at the station, event days removed, 50th percentile. For a multi-day episode, the 3 days before its start and the 3 days after its end (without event days) are the reference for every day of the episode.
* **TH stations** never self-calibrate. They use the Option 1 load of their RB or, without a suitable RB, the Option 2 load of their paired UB.

With fewer valid reference days than `MIN_REFERENCE_DAYS_OPTION1` / `MIN_REFERENCE_DAYS_OPTION2`, the day is flagged `insufficient_reference` and nothing is subtracted.

The contribution is subtracted on **every** event day, with or without an exceedance; other days get 0. `adjusted_value = measured − natural_contribution`, never below 0. A dust day is confirmed (`confirmed_dust_day`) only when the quantified contribution is > 0.

New daily columns: `station_class`, `method`, `reference_station_id`, `reference_value`, `n_reference_days`, `natural_contribution`, `cams_dust_flag`, `confirmed_dust_day`, `adjusted_value`, `insufficient_reference`, `pairing_source`. The previous names are kept for the plots: `dust_flag` = `cams_dust_flag`, `pollutant_median` = `reference_value`, `Dust_contribution` = `natural_contribution`, `corrected_pollutant` = `adjusted_value`.

**Model-specific dust threshold.** `dust_threshold_sensitivity.natural_threshold_sweep`, `contribution_consistency` and `lowest_consistent_threshold` recompute the deduction for a list of CAMS thresholds and report, per country or region, the lowest threshold from which the quantified contribution is > 0 on at least a given share of CAMS dust days. The notebook runs this in section 7.2.1 when `RUN_THRESHOLD_SWEEP = True`.

An alternative formula that scales CAMS dust by the ratio of observed PM to CAMS PM is sketched in the notebook and left commented out. It is not used.

### 6. Outputs

When `LOAD_PARQUET_DATA` is `False`, the daily table for all stations is written as a Parquet file in the output folder (Google Drive or the project folder):

`CAMS_dust_{POLLUTANT}_deduction_{dataset}_{EEA_temporal_flag}_{YEAR}_MA{Basline_MA_days}.parquet`

Example: `CAMS_dust_PM10_deduction_E2a_hour_2025_MA6.parquet`. A later run with `LOAD_PARQUET_DATA = True` reads that same path and skips sections 3–6. With `REAPPLY_PAIRING_ON_LOAD = True` the deduction is recomputed on the loaded table with the current pairing; that table covers only the selected year, so the first and last days have a shorter reference window.

The adjusted daily series is summarised per station for the selected year (`natural_contribution.station_indicators`). Exceedances and annual means are recalculated from the daily values, with and without the natural contribution:

| Column | Meaning |
|---|---|
| `station_class`, `method`, `reference_station_id`, `pairing_source` | Class, method and reference station used; default or manual pairing |
| `cams_dust_days`, `confirmed_dust_days`, `insufficient_reference_days` | CAMS dust days, dust days with a contribution > 0, days without enough reference data |
| `exceedances_measured`, `exceedances_adjusted` | Days above the daily limit before and after subtraction |
| `exceedances_natural` | Exceedances attributed to natural sources |
| `annual_mean_measured`, `annual_mean_adjusted` | Annual mean before and after subtraction |
| `annual_limit_exceeded_measured`, `annual_limit_exceeded_adjusted` | Annual mean above the annual limit |
| `Exceedance_days`, `Dust_exceedance_days`, `NonDust_exceedance`, `Average_pollutant`, `Average_pollutant_dust_removed` | Previous names, same meaning, used by the maps |

The station summary is written as CSV in the output folder:

* one country: `{country code}_station_annual_{POLLUTANT}_{dataset}_{EEA_temporal_flag}_{YEAR}_MA{Basline_MA_days}.csv` (for example `ES_station_annual_PM10_E2a_hour_2025_MA6.csv`)
* two or more countries: `muti_countries_station_annual_...csv` (the prefix is spelled that way in the notebook)

### 7. Visualisation

Interactive figures are aimed at Google Colab. If the widgets do not run (for example in VS Code or on an HPC node), use the static plots.

* **Interactive annual map** (Plotly): colour is the annual mean after dust removal; marker size is the number of exceedance days that are not attributed to dust.
* **Interactive station time series** (`map_timeseries_clickable_plot` in `util.py`, ipyleaflet): click a station to plot measured and corrected concentrations, with CAMS dust and the two thresholds.
* **Static time series** (`plot_station_timeseries`): same figure for a station id you type in.
* **Static maps** (`plot_exceedance_maps_discrete`, Cartopy): maps of any station-summary column. An *Edit here* block sets `columns_to_plot`, titles, and the colour scale. The map is limited to the countries selected in `Countries`.

## Other Python files

The notebook downloads a helper module from `REPO_RAW_BASE` only when the file is not already next to the notebook, so local edits are kept. In Colab, point `REPO_RAW_BASE` at a public repository that contains `pairing.py`, `natural_contribution.py` and `pairing_viz.py`, or upload those files to the Colab session.

### `pairing.py`

Station classification (`classify_station`, `build_station_table`), default pairing (`default_pairing`, `PairingLimits`), method rules (`derive_method`, `resolve_method`), manual edits (`set_manual_pair`, `candidate_references`, `reset_station`, `reset_all`) and the CSV round trip (`export_pairing_csv`, `read_pairing_csv`, `apply_user_pairing`, `load_pairing`). Pure pandas/numpy.

### `natural_contribution.py`

Window selection (`reference_window`, `reference_for_days`, `episode_bounds`), the per-station net load (`station_net_load`), the deduction for all stations (`compute_natural_contribution`, `run_natural_deduction`), limit values (`LimitValues`), method parameters (`MethodParams`) and the station summary (`station_indicators`). Pure pandas/numpy.

### `pairing_viz.py`

Plotly and Cartopy pairing maps, `PairingSession` (the pairing table the deduction reads, with listeners that redraw the map), the ipywidgets map and editor (`pairing_app`), and Colab download/upload helpers.

### Tests

`python -m unittest discover -s tests` (with the repository root on `PYTHONPATH`). `tests/test_natural_pairing.py` covers window selection with excluded event days, multi-day episodes, pairing by distance and altitude, TH stations using the UB load, the annual mean over all dust days, the threshold helper and the pairing CSV round trip with validation errors.

The remaining scripts are not imported by the notebook, except `util.py`, `cams_data_to_eea_daily.py` and, for section 7.2.1, `dust_threshold_sensitivity.py`. Paths inside the standalone scripts still point at a TNO project directory unless you edit `project_dir`.

### `util.py`

Shared helpers. The notebook downloads this file from the `main` branch at the start of a run and imports it. The functions the notebook calls are:

| Function | What it does |
|---|---|
| `until_check` | Returns a short string so you can see that the import worked |
| `plot_station_pairing` | Pairing map from `pairing_viz.py`: Plotly by default, Cartopy with `static=True` |
| `filter_daily_by_coverage` | Keeps every row for stations whose coverage in a reference year is at least `min_pct` (so days just outside that year remain available for the background window) |
| `add_cams_daily_dust_by_station` | Interpolates a CAMS daily field to each station and attaches `cams_dust` |
| `compute_station_baseline` | Previous method: for dust days that are also exceedances, median of `neighbor_n` non-dust days before and after. No longer used by the notebook |
| `plot_station_timeseries` | Measured vs corrected concentration and CAMS dust for one station |
| `map_timeseries_clickable_plot` | ipyleaflet map; clicking a marker draws that station’s time series |
| `plot_exceedance_maps_discrete` | Static Cartopy maps of station-summary columns |
| `plot_interactive_station_map` | Plotly Mapbox scatter of a station table |
| `eea_hourly_to_utc` | Treats naive EEA timestamps as fixed UTC+1 and converts them to UTC |
| `calculate_data_coverage` | Coverage of each station between two dates (used by the daily-data branch) |
| `compute_median_for_station` | Same background idea as `compute_station_baseline`, with a fixed window of 15 non-dust days on each side |

`filter_daily_by_coverage` and `add_cams_daily_dust_by_station` are each defined twice in the file. The later definition is the one that is used. That coverage function uses the `calendar` module, which `util.py` does not import itself; the notebook assigns `util.calendar = calendar` before calling it. The notebook also pastes its own copy of `plot_exceedance_maps_discrete` into the static-map section.

### `cams_data_download.py`

Standalone CAMS download, separate from the notebook’s download cells.

* **Inputs:** Copernicus ADS credentials in `~/.cdsapirc`. `YEAR` (default 2024) and `VAR` (default `dust`) are set at the top.
* **What it does:** Requests `cams-europe-air-quality-reanalyses` (ensemble, surface level, interim reanalysis) for December of the previous year and for each quarter of `YEAR`, downloads five zip files, and extracts them.
* **Outputs:** `./wp-dust/IRA_dust/CAMS_IRA_*.zip` and the extracted NetCDF files in that folder.
* **Relation to the notebook:** Same ADS dataset and the same five-file split. The notebook writes to `IRA_dust/` (or the Google Drive copy of that folder) instead of `wp-dust/IRA_dust/`.

### `cams_data_to_eea_daily.py`

Daily-observation counterpart of `Dust_discount_hourly.py`. The notebook calls it when `EEA_temporal_flag` is `day`. Importing the module does not download or compute; `python cams_data_to_eea_daily.py` runs `main()`.

* **Inputs:** The same settings as the notebook, at the top of the file: `Countries`, `YEAR`, `POLLUTANT` (`PM10` or `PM2.5`), `dataset` (`E1a` or `E2a`), `EEA_temporal_flag = 'day'`, `DOWNLOAD_EEA`, `DOWNLOAD_CAMS`, `COMPUTE_CAMS_DAILY`, `USE_GOOGLE_DRIVE`, dust threshold, coverage (default 65%), and `Basline_MA_days` (default 6). `resolve_run_flags` applies the same rules as the notebook: `DOWNLOAD_CAMS` turns `COMPUTE_CAMS_DAILY` on. EEA data come from the same download API as the notebook (dataset id 1 for `E2a`, 2 for `E1a`, 14 December of the previous year through 31 December, `aggregationType` `day`). An existing zip is extracted when `DOWNLOAD_EEA` is `False`. CAMS zips are stored in the output folder and extracted only when the daily product is rebuilt. Station coordinates and `Timezone` come from `DataExtract.csv` in the output folder, or from `Data/DataExtract.csv` if that is the copy you have.
* **What it does:** Filters daily EEA values the way the notebook filters hourly values (validity, verification, non-negative), without converting local dates to UTC. Keeps stations that meet the coverage rule. Builds CAMS daily means on each station’s local day, in memory-safe chunks, and reuses Drive or local Parquet caches. Then applies the same dust flag (`cams_dust` above the threshold), exceedance flag, neighbour-median background, and clipped dust contribution as the notebook. The background is computed only on days that are both dusty and an exceedance, matching `util.compute_station_baseline` (the hourly script still has the exceedance term commented out).
* **Output:** `CAMS_dust_{POLLUTANT}_deduction_{dataset}_day_{YEAR}_MA{Basline_MA_days}.parquet` in the Google Drive output folder or the project folder. Columns match the notebook (`day`, `daily_mean`, `cams_dust`, `dust_flag`, `Exceedance`, `pollutant_median`, `Dust_contribution`, `corrected_pollutant`, coordinates), so `LOAD_PARQUET_DATA = True` reads this file the same way it reads an hourly result. The filename includes the time resolution, so it does not overwrite an hourly run.
* **Relation to the notebook:** Section 4 of the notebook calls `load_and_filter_eea_daily` and `add_cams_local_daily_dust`, then continues with the existing deduction, summary, and plots. Running the script by itself performs that whole chain and writes the Parquet the notebook would load. Section 7.3 calls `load_and_combine_deduction_parquets` when you choose to plot the hourly and daily tables together.

### `dust_threshold_sensitivity.py`

Recomputes the hourly deduction at CAMS dust thresholds 3 through 10 µg/m³ and compares exceedance days.

* **Inputs:** The saved hourly deduction Parquet (`CAMS_dust_{POLLUTANT}_deduction_{dataset}_hour_{YEAR}_MA{n}.parquet`), `--pollutant` (`PM10` or `PM2.5`), and `--neighbor-n` (default 6, the notebook’s `Basline_MA_days`). The default thresholds are 3, 4, 5, 6, 7, 8, 9 and 10 µg/m³. The reference for the scatter plots is 5 µg/m³.
* **What it does:** For each threshold, flags dust days where `cams_dust` is above that value and rebuilds the neighbour-median background, dust contribution, and corrected concentration the way the notebook does. Plots and the summary use exceedance days, the days when the measured daily mean is above the pollutant limit. The file is the hourly product, so `day` is a UTC day. Neighbour days are taken from that saved year. The notebook also uses 14 December of the previous year, so the first days of January can differ slightly from a full notebook run.
* **Outputs:** In `dust_threshold_sensitivity/` (or `--output-dir`): `threshold_summary.csv`, `scatter_contribution_exceedance.png` (contribution at each threshold against 5 µg/m³), `scatter_dust_exceedance_days.png` (per station, how many exceedance days are attributed to dust), `distribution_contribution_exceedance.png` (boxplots), and `distribution_corrected_exceedance.png` (overlaid histograms of the corrected concentration).
* **Relation to the notebook:** It does not change the notebook result. Run it after an hourly deduction Parquet exists: `python dust_threshold_sensitivity.py --parquet CAMS_dust_PM10_deduction_E2a_hour_2024_MA6.parquet`.

### `Dust_discount_hourly.py`

Standalone hourly PM10 deduction, with settings fixed in the script rather than in notebook widgets.

* **Inputs:** EEA hourly Parquet under `{project_dir}/EEA_PM10/E1a/hour`, metadata `{project_dir}/EEA_PM10/DataExtract.csv`, and either raw CAMS NetCDF in `IRA_dust/` or an existing `IRA_dust/cams_dust_daily_mean.nc`. Optional flags `DOWNLOAD_EEA`, `DOWNLOAD_CAMS`, and `COMPUTE_CAMS_DAILY` default to `False`. The country list, year (2024), dust threshold (5 µg/m³), and PM10 limit (50 µg/m³) are hard-coded. `project_dir` defaults to a TNO path.
* **What it does:** Same outline as the notebook’s hourly path: optional EEA and CAMS downloads, CET-to-UTC conversion, validity and verification filters, 24-hour daily means, a coverage filter (here 75% in 2024), metadata join, interpolation of daily CAMS dust, a dust flag, and a background median. The background uses 15 non-dust days before and after each dust day, and dust is flagged with `>= 5 µg/m³`. Negative contributions are clipped to zero. Corrected PM10 on dust days is the measurement minus that contribution.
* **Output:** `{project_dir}CAMS_dust_deduction_{dataset}_{EEA_temporal_flag}_{YEAR}_v2.parquet`.
* **Relation to the notebook:** An earlier, non-interactive version of the hourly calculation. The notebook does not call it and does not look for the `_v2` filename. You can still open the Parquet in the notebook if you point the load cell at it.

### `storage_flowchart.py`

Draws two Matplotlib figures of an earlier description of storage and processing. Running it writes, in the current directory:

* `storage_flowchart.png` — download, filter, dust flag, baseline, and outputs
* `storage_hierarchy.png` — folder layout and a short list of helper functions

It does not read EEA or CAMS data and is not used by the notebook. The numbers on those figures (for example 75% coverage and a 30-day or 15-day window) are the ones that were drawn into the script; they are not the notebook defaults (65% coverage and `Basline_MA_days = 6`). The workflow chart in `docs/` is a separate image of the current notebook.

## How to run

### Google Colab

1. Open `Dust_deduction_tool_full.ipynb` in Colab (the first notebook cell links to the copy on GitHub).
2. Run Section 1. It installs `cartopy` and replaces the local `util.py` with the copy on the `main` branch. The daily path also needs `cams_data_to_eea_daily.py` in the working directory (clone the repository, or upload the file). If that file is missing, the daily cell downloads whatever is on `main`.
3. In Section 2, set countries, year, pollutant, dataset, and the run flags. Leave `USE_GOOGLE_DRIVE = True` unless you have another place to store the files. For EEA daily data set `EEA_temporal_flag = 'day'`. Leave it as `hour` to keep the UTC daily-mean path. `DOWNLOAD_CAMS = True` turns `COMPUTE_CAMS_DAILY` on. A download or a new daily product also turns `LOAD_PARQUET_DATA` off, and the cell prints that change.
4. Put `DataExtract.csv` in `MyDrive/CAMS_Tool_output` before the metadata step.
5. For a first CAMS download, create an ADS account, accept the dataset licence, and paste the API key when the notebook asks.
6. Run the notebook from top to bottom. Set `DOWNLOAD_EEA`, `DOWNLOAD_CAMS`, and `COMPUTE_CAMS_DAILY` back to `False`, and `LOAD_PARQUET_DATA` to `True`, when you only want to reload results and redraw figures. Leaving a download flag on forces the load flag off.
7. Section 7.3 can stack the hourly and daily Parquet files into one table for the Section 8 plots. The markdown cell above that switch is the time-zone disclaimer: the hourly `day` is a UTC day and the daily `day` is the station’s local day, so the same date is not the same 24 hours. Leave `COMBINE_HOURLY_DAILY_FOR_VIZ = False` to plot only the resolution from Section 2.
8. After an hourly Parquet exists, `python dust_threshold_sensitivity.py --parquet <hourly file>` compares dust thresholds from 3 to 10 µg/m³ on exceedance days.

Interactive maps are written for Colab. If a widget fails, use the static time series and static maps in Section 8.

### Local

1. Clone the repository and install the packages listed below.
2. In Section 2 set `USE_GOOGLE_DRIVE = False`. Outputs then go to the project directory (`project_dir = '.'`).
3. Copy `Data/DataExtract.csv` to `./DataExtract.csv`, or download a newer extract and save it there. The notebook does not read `Data/DataExtract.csv` by itself.
4. Run the cells in order. Skip or adapt the Colab-only lines (`google.colab`, Drive mount). Section 1 still tries to download `util.py` from GitHub; if you are editing that file locally, skip the download and import the local module instead.
5. The clickable ipyleaflet map may not work outside Colab or Jupyter. The static plots in Section 8 do not need it.
6. `cams_data_download.py`, `cams_data_to_eea_daily.py`, and `Dust_discount_hourly.py` are run as scripts (`python cams_data_download.py`, and likewise for the others) after you set `project_dir` and the year. They expect a `~/.cdsapirc` file when they download CAMS data.

## Requirements

There is no pinned environment file. The notebook and scripts import:

* Notebook and `util.py`: `numpy`, `pandas`, `matplotlib`, `cartopy`, `xarray`, `netCDF4`, `scipy`, `plotly`, `pyarrow`, `requests`, `ipywidgets`, `ipyleaflet`
* CAMS download (notebook and `cams_data_download.py`): `cdsapi` (>= 0.7.7)
* `cams_data_to_eea_daily.py`: also `psutil`
* Colab only: `google.colab` (Drive mount)

`storage_flowchart.py` needs only `numpy` and `matplotlib`.

## Outputs

| Product | Where | Name |
|---|---|---|
| EEA zip | Google Drive `EEA_{POLLUTANT}/{temporal}/` or the local EEA folder | `{dataset}_{POLLUTANT}_{temporal}_{YEAR}.zip` |
| CAMS zips | `IRA_dust/` on Drive, or the project folder | `CAMS_IRA_*.zip` (the stored download) |
| CAMS hourly NetCDF | Local `IRA_dust/` only, and only while computing | `cams.eaq.ira.ENSa.dust*.nc` |
| Daily CAMS field (hourly path, UTC) | `IRA_dust/` | `cams_dust_{YEAR}_daily_mean.nc` |
| Station-hourly CAMS cache (daily path) | `IRA_dust/` on Drive or in the project folder | `cams_dust_station_hourly_<stations>_<source>.parquet` |
| Local-day CAMS cache (daily path) | `IRA_dust/` on Drive or in the project folder | `cams_dust_local_daily_stations_<stations>_<year>_h18.parquet` |
| Daily deduction table | Google Drive output folder | `CAMS_dust_{POLLUTANT}_deduction_{dataset}_{temporal}_{YEAR}_MA{n}.parquet` |
| Daily deduction table (local save) | Project folder | `CAMS_dust_deduction_{dataset}_{temporal}_{YEAR}.parquet` |
| Station summary | Same output folder | `{country}_station_annual_....csv` or `muti_countries_station_annual_....csv` |
| Dust-threshold comparison | `dust_threshold_sensitivity/` | `threshold_summary.csv` and the scatter and distribution PNGs from `dust_threshold_sensitivity.py` |

Figures are shown in the notebook. The static-map function can also write a PNG if you pass `savefile`.

## Known limitations / To do

Daily CAMS dust is now part of the deduction when `EEA_temporal_flag` is `day`. A few limits remain:

* **Daylight-saving time is not modelled.** Both paths use the fixed offset in the data: UTC+1 for every hourly stamp, and the `Timezone` label for each daily station. If a country builds its official daily value in civil time, the hour of a spring or autumn transition can sit in the neighbouring local day.
* **The hourly path’s UTC grid can still exceed Colab memory.** `COMPUTE_CAMS_DAILY` on `EEA_temporal_flag = 'hour'` still averages the full CAMS domain. The daily path avoids that by sampling stations one UTC day at a time. The first compute still extracts the hourly NetCDF from the stored zips, and a Copernicus ADS key is required to download a missing zip.
* **A few setup mismatches are still in the notebook.** `DOWNLOAD_EEA = True` deletes and replaces an existing EEA zip; `DOWNLOAD_CAMS = True` reuses a zip that is already in the output folder. With `USE_GOOGLE_DRIVE = False`, the save cell writes `CAMS_dust_deduction_{dataset}_{temporal}_{YEAR}.parquet` (and `project_dir = '.'` has no slash), while the load cell looks for the longer `CAMS_dust_{POLLUTANT}_..._MA{n}` name. The hourly UTC-grid step only globs NetCDF files whose name contains `YEAR`, so December of the previous year is missing on that path; the local-day path includes it. Section 1 always replaces `util.py` with the copy on `main`. `DataExtract.csv` is still a file you place by hand. Extracted hourly NetCDF is left on the Colab disk after the run so later cells can read it; it is not uploaded to Drive.
* **Stations outside the CAMS European domain are dropped** on both paths (for example some overseas territories whose metadata offset is `UTC-04` or `UTC+04`).
* **A local day is kept only when at least 75% of its CAMS hours are present** (18 hours for hourly files). The first and last days of the archive can be missing for stations whose offset reaches outside the downloaded hours.
* The workflow chart in `docs/` shows the hourly path. The daily branch is the extra step in section 4.5 of the notebook.
* **A combined hourly and daily plot compares date labels, not one physical day.** Section 7.3 states this before the merge. The hourly product is a UTC day. The daily product is the station’s local day.
