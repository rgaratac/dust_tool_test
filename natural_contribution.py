"""
Natural-contribution deduction following the EU methodology for natural sources.

Event days are the days flagged as natural events (for now: CAMS dust above
``dust_threshold`` at the station). Event days are excluded from every
percentile window. The contribution is subtracted on every event day, with or
without an exceedance.

Option 1 (regional background)
    At the paired RB station, for each target day d: window d-15 .. d+15,
    event days and d itself dropped, 50th percentile of the rest. The net load
    max(RB(d) - reference, 0) is subtracted at every station paired to that RB
    (including the RB itself).

Option 2 (self-calibration, UB without suitable RB)
    At the UB station: window d-3 .. d+3, event days dropped, 50th percentile.
    A multi-day episode uses the days before its start and after its end,
    (start-3 .. start-1 and end+1 .. end+3, event days dropped) for every day
    of the episode.

TH stations never self-calibrate. They take the Option 1 load of their RB or,
without a suitable RB, the Option 2 load of their paired UB.

The pairing table (``pairing.py``) decides the method and reference station
for each station, so editing the pairing changes the next run directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MethodParams:
    option1_half_window: int = 15
    option2_half_window: int = 3
    percentile: float = 50.0
    min_reference_days_option1: int = 10
    min_reference_days_option2: int = 3


@dataclass(frozen=True)
class LimitValues:
    """Daily and annual limit values, µg/m³ (Directive (EU) 2024/2881 defaults)."""

    pm10_daily: float = 45.0
    pm25_daily: float = 25.0
    pm10_annual: float = 20.0
    pm25_annual: float = 10.0

    def daily(self, pollutant: str) -> float:
        return self.pm10_daily if _is_pm10(pollutant) else self.pm25_daily

    def annual(self, pollutant: str) -> float:
        return self.pm10_annual if _is_pm10(pollutant) else self.pm25_annual


def _is_pm10(pollutant: str) -> bool:
    p = str(pollutant).upper().replace(".", "")
    if p == "PM10":
        return True
    if p == "PM25":
        return False
    raise ValueError(f"Pollutant {pollutant} not supported. Must be PM10 or PM2.5.")


NEW_COLUMNS = [
    "station_class", "method", "reference_station_id", "reference_value",
    "n_reference_days", "natural_contribution", "cams_dust_flag",
    "confirmed_dust_day", "adjusted_value", "insufficient_reference", "pairing_source",
]


def calendar_day(series: pd.Series) -> pd.Series:
    """Naive calendar day without moving the clock (aware stamps drop their tz)."""
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.floor("D").astype("datetime64[ns]")


# ---------------------------------------------------------------------------
# Window selection on a contiguous daily grid
# ---------------------------------------------------------------------------

def episode_bounds(events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last index of the consecutive event run at each position (-1 elsewhere)."""
    events = np.asarray(events, dtype=bool)
    n = events.size
    start = np.full(n, -1)
    end = np.full(n, -1)
    i = 0
    while i < n:
        if events[i]:
            j = i
            while j + 1 < n and events[j + 1]:
                j += 1
            start[i : j + 1] = i
            end[i : j + 1] = j
            i = j + 1
        else:
            i += 1
    return start, end


def reference_window(
    values: np.ndarray,
    events: np.ndarray,
    target: int,
    *,
    half_window: int,
    use_episode: bool,
    episode: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Values that enter the percentile for one target index.

    Event days and the target day are always dropped. With ``use_episode`` the
    window is built around the whole event run that contains the target.
    """
    values = np.asarray(values, dtype=float)
    events = np.asarray(events, dtype=bool)
    lo_day = hi_day = target
    if use_episode and events[target]:
        start, end = episode if episode is not None else episode_bounds(events)
        lo_day, hi_day = start[target], end[target]
    lo = max(0, lo_day - half_window)
    hi = min(values.size, hi_day + half_window + 1)
    keep = ~events[lo:hi] & ~np.isnan(values[lo:hi])
    keep[target - lo] = False
    return values[lo:hi][keep]


def reference_for_days(
    values: np.ndarray,
    events: np.ndarray,
    targets,
    *,
    half_window: int,
    percentile: float,
    min_valid: int,
    use_episode: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile reference and number of days used, NaN when below ``min_valid``."""
    episode = episode_bounds(events) if use_episode else None
    targets = np.asarray(targets, dtype=int)
    ref = np.full(targets.size, np.nan)
    n_used = np.zeros(targets.size, dtype=int)
    for k, t in enumerate(targets):
        pool = reference_window(
            values, events, int(t), half_window=half_window,
            use_episode=use_episode, episode=episode,
        )
        n_used[k] = pool.size
        if pool.size >= max(1, min_valid):
            ref[k] = np.percentile(pool, percentile)
    return ref, n_used


def _grid(days: pd.Series, values: pd.Series, events: pd.Series, extra_days=()):
    """Contiguous daily grid covering the station's days and any extra target days."""
    frame = pd.DataFrame({"d": days.to_numpy(), "v": values.to_numpy(dtype=float), "e": events.to_numpy(dtype=bool)})
    frame = frame.dropna(subset=["d"]).groupby("d").agg(v=("v", "mean"), e=("e", "max"))
    all_days = list(frame.index) + list(extra_days)
    if not all_days:
        return pd.DatetimeIndex([]), np.array([]), np.array([], dtype=bool)
    index = pd.date_range(min(all_days), max(all_days), freq="D")
    frame = frame.reindex(index)
    return index, frame["v"].to_numpy(dtype=float), frame["e"].fillna(False).to_numpy(dtype=bool)


def station_net_load(
    days: pd.Series,
    values: pd.Series,
    events: pd.Series,
    target_days,
    *,
    mode: str,
    params: MethodParams = MethodParams(),
) -> pd.DataFrame:
    """Net natural load at one reference station for the requested days.

    ``mode`` is ``option1`` (±15 days, per day) or ``option2`` (±3 days,
    episode-aware). Returns one row per target day with ``reference_value``,
    ``n_reference_days``, ``load`` and ``insufficient_reference``.
    """
    target_days = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(target_days)))))
    index, vals, evts = _grid(days, values, events, extra_days=target_days)
    if mode == "option1":
        half, min_valid, episodes = params.option1_half_window, params.min_reference_days_option1, False
    elif mode == "option2":
        half, min_valid, episodes = params.option2_half_window, params.min_reference_days_option2, True
    else:
        raise ValueError("mode must be 'option1' or 'option2'")
    pos = index.get_indexer(target_days)
    ref, n_used = reference_for_days(
        vals, evts, pos, half_window=half, percentile=params.percentile,
        min_valid=min_valid, use_episode=episodes,
    )
    measured = vals[pos] if len(pos) else np.array([])
    load = np.maximum(measured - ref, 0.0)
    return pd.DataFrame(
        {
            "reference_value": ref,
            "n_reference_days": n_used,
            "load": load,
            "insufficient_reference": np.isnan(load),
        },
        index=target_days,
    )


# ---------------------------------------------------------------------------
# Deduction for all stations
# ---------------------------------------------------------------------------

def compute_natural_contribution(
    df: pd.DataFrame,
    pairing: pd.DataFrame,
    *,
    dust_threshold: float,
    params: MethodParams = MethodParams(),
    daily_limit: float | None = None,
    station_col: str = "Samplingpoint",
    time_col: str = "day",
    value_col: str = "daily_mean",
    cams_col: str = "cams_dust",
) -> pd.DataFrame:
    """Add the natural-contribution columns to a daily station table.

    The legacy columns used by the notebook plots are filled too:
    ``dust_flag`` = ``cams_dust_flag``, ``pollutant_median`` =
    ``reference_value``, ``Dust_contribution`` = ``natural_contribution`` and
    ``corrected_pollutant`` = ``adjusted_value``.
    """
    work = df.drop(columns=[c for c in NEW_COLUMNS if c in df.columns])
    work["_d"] = calendar_day(work[time_col])
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").astype("float32")
    cams = pd.to_numeric(work[cams_col], errors="coerce")
    work["cams_dust_flag"] = (cams > float(dust_threshold)).fillna(False).astype(bool)

    pair = pairing.copy()
    pair["station_id"] = pair["station_id"].astype(str)
    pair = pair.drop_duplicates("station_id").set_index("station_id")
    sid = work[station_col].astype(str)
    method = sid.map(pair["method"]).fillna("none")
    unpaired = sorted(set(sid[~sid.isin(pair.index)]))
    if unpaired:
        print(f"Warning: {len(unpaired)} station(s) are not in the pairing table; nothing is subtracted there.")
    work["station_class"] = sid.map(pair["class"]).fillna("unknown")
    work["method"] = method
    work["pairing_source"] = sid.map(pair["source"]).fillna("default")
    ref_id = pd.Series(None, index=work.index, dtype=object)
    ref_id[method == "option1"] = sid.map(pair["paired_rb_id"])[method == "option1"]
    ref_id[method == "option2"] = sid[method == "option2"]
    ref_id[method == "option2_ub_paired"] = sid.map(pair["paired_ub_id"])[method == "option2_ub_paired"]
    work["reference_station_id"] = ref_id
    mode = np.where(method == "option1", "option1", "option2")
    work["_mode"] = np.where(method == "none", "", mode)

    needs = work[work["cams_dust_flag"] & ref_id.notna() & (work["_mode"] != "")]
    station_groups = {str(k): g for k, g in work.groupby(sid, sort=False)}
    records = []
    for (ref, m), grp in needs.groupby(["reference_station_id", "_mode"], sort=False):
        targets = grp["_d"].dropna().unique()
        ref_rows = station_groups.get(str(ref))
        if ref_rows is None:
            res = pd.DataFrame(
                {"reference_value": np.nan, "n_reference_days": 0, "load": np.nan, "insufficient_reference": True},
                index=pd.DatetimeIndex(targets),
            )
        else:
            res = station_net_load(
                ref_rows["_d"], ref_rows[value_col], ref_rows["cams_dust_flag"], targets,
                mode=m, params=params,
            )
        res["reference_station_id"] = ref
        res["_mode"] = m
        records.append(res.rename_axis("_d").reset_index())

    cols = ["reference_station_id", "_mode", "_d", "reference_value", "n_reference_days", "load", "insufficient_reference"]
    loads = pd.concat(records, ignore_index=True)[cols] if records else pd.DataFrame(columns=cols)
    loads["reference_station_id"] = loads["reference_station_id"].astype(object)
    loads["_d"] = pd.to_datetime(loads["_d"]).astype("datetime64[ns]")
    original_index = work.index
    work = work.reset_index(drop=True).merge(loads, on=["reference_station_id", "_mode", "_d"], how="left")
    work.index = original_index

    event = work["cams_dust_flag"]
    measured = work[value_col].astype(float)
    load = work["load"].astype(float)
    has_load = event & load.notna()
    contribution = np.where(has_load, np.minimum(load, measured), 0.0)
    contribution = np.where(np.isnan(contribution), 0.0, contribution)
    work["natural_contribution"] = contribution.astype("float32")
    work["adjusted_value"] = np.maximum(measured - contribution, 0.0).astype("float32")
    work["reference_value"] = work["reference_value"].where(event).astype("float32")
    work["n_reference_days"] = work["n_reference_days"].where(event).fillna(0).astype("int16")
    work["insufficient_reference"] = (event & (work["method"] != "none") & load.isna()).astype(bool)
    work["confirmed_dust_day"] = (event & (work["natural_contribution"] > 0)).astype(bool)

    work["dust_flag"] = work["cams_dust_flag"]
    work["pollutant_median"] = work["reference_value"]
    work["Dust_contribution"] = work["natural_contribution"]
    work["corrected_pollutant"] = work["adjusted_value"]
    if daily_limit is not None:
        work["Exceedance"] = measured > daily_limit
        work["adjusted_exceedance"] = work["adjusted_value"] > daily_limit
        work["Dust_exceedance"] = work["Exceedance"] & ~work["adjusted_exceedance"]
    return work.drop(columns=["_d", "_mode", "load"])


def run_natural_deduction(
    df: pd.DataFrame,
    pairing: pd.DataFrame,
    year: int,
    *,
    pollutant: str,
    dust_threshold: float,
    params: MethodParams = MethodParams(),
    limits: LimitValues = LimitValues(),
    time_col: str = "day",
) -> pd.DataFrame:
    """Deduction over the full table (incl. December margin), then the target year."""
    out = compute_natural_contribution(
        df, pairing, dust_threshold=dust_threshold, params=params,
        daily_limit=limits.daily(pollutant), time_col=time_col,
    )
    day = calendar_day(out[time_col])
    target = out[(day >= pd.Timestamp(f"{year}-01-01")) & (day < pd.Timestamp(f"{year + 1}-01-01"))].copy()
    if target.empty and not out.empty:
        print(f"Warning: no rows in {year}; returning the full table.")
        target = out
    return target


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def _first_valid(series: pd.Series):
    s = series.dropna()
    return s.mode().iloc[0] if len(s) else None


def station_indicators(
    daily: pd.DataFrame,
    *,
    daily_limit: float,
    annual_limit: float,
    station_col: str = "Samplingpoint",
    value_col: str = "daily_mean",
) -> pd.DataFrame:
    """Exceedances and annual means with and without the natural contribution.

    The legacy summary columns (``Exceedance_days``, ``Dust_exceedance_days``,
    ``NonDust_exceedance``, ``Average_pollutant``,
    ``Average_pollutant_dust_removed``) are kept for the existing maps.
    """
    work = daily.copy()
    measured = pd.to_numeric(work[value_col], errors="coerce")
    adjusted = pd.to_numeric(work["adjusted_value"], errors="coerce")
    work["_exc_meas"] = measured > daily_limit
    work["_exc_adj"] = adjusted > daily_limit
    work["_measured"] = measured
    work["_adjusted"] = adjusted
    agg = {
        "n_days": ("_measured", "count"),
        "cams_dust_days": ("cams_dust_flag", "sum"),
        "confirmed_dust_days": ("confirmed_dust_day", "sum"),
        "insufficient_reference_days": ("insufficient_reference", "sum"),
        "exceedances_measured": ("_exc_meas", "sum"),
        "exceedances_adjusted": ("_exc_adj", "sum"),
        "annual_mean_measured": ("_measured", "mean"),
        "annual_mean_adjusted": ("_adjusted", "mean"),
        "station_class": ("station_class", "first"),
        "method": ("method", "first"),
        "reference_station_id": ("reference_station_id", _first_valid),
        "pairing_source": ("pairing_source", "first"),
    }
    for col in ("Latitude", "Longitude", "Altitude"):
        if col in work.columns:
            agg[col] = (col, "first")
    table = work.groupby(station_col, as_index=False, observed=True).agg(**agg)
    table["exceedances_natural"] = table["exceedances_measured"] - table["exceedances_adjusted"]
    table["annual_limit_exceeded_measured"] = table["annual_mean_measured"] > annual_limit
    table["annual_limit_exceeded_adjusted"] = table["annual_mean_adjusted"] > annual_limit

    table["Exceedance_days"] = table["exceedances_measured"]
    table["Dust_exceedance_days"] = table["exceedances_natural"]
    table["NonDust_exceedance"] = table["exceedances_adjusted"]
    table["Average_pollutant"] = table["annual_mean_measured"]
    table["Average_pollutant_dust_removed"] = table["annual_mean_adjusted"]

    order = [station_col, "Latitude", "Longitude", "Altitude", "station_class", "method",
             "reference_station_id", "pairing_source", "n_days", "cams_dust_days",
             "confirmed_dust_days", "insufficient_reference_days", "exceedances_measured",
             "exceedances_adjusted", "exceedances_natural", "annual_mean_measured",
             "annual_mean_adjusted", "annual_limit_exceeded_measured",
             "annual_limit_exceeded_adjusted", "Exceedance_days", "Dust_exceedance_days",
             "NonDust_exceedance", "Average_pollutant", "Average_pollutant_dust_removed"]
    return table[[c for c in order if c in table.columns]]
