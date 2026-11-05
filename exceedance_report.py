"""Tables and plots on how the natural (dust) deduction changes exceedances.

Everything is computed from the daily deduction table (``df_target_year``), so
it works for freshly computed results, a loaded Parquet file and the combined
hourly/daily table. Columns used: ``Samplingpoint``, ``day``, ``daily_mean``,
``corrected_pollutant`` (adjusted value), ``Dust_contribution`` and
``dust_flag``; ``station_class`` is used when present.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

STATION = "Samplingpoint"
MEASURED = "daily_mean"
ADJUSTED = "corrected_pollutant"
CONTRIBUTION = "Dust_contribution"
DUST_FLAG = "dust_flag"

MEASURED_COLOR = "#d62728"
ADJUSTED_COLOR = "#1f77b4"
NATURAL_COLOR = "#e6a23c"


def _prepare(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    work["day"] = pd.to_datetime(work["day"], errors="coerce", utc=True).dt.tz_localize(None)
    work[MEASURED] = pd.to_numeric(work[MEASURED], errors="coerce")
    work[ADJUSTED] = pd.to_numeric(work[ADJUSTED], errors="coerce")
    if CONTRIBUTION not in work.columns:
        work[CONTRIBUTION] = work[MEASURED] - work[ADJUSTED]
    work[CONTRIBUTION] = pd.to_numeric(work[CONTRIBUTION], errors="coerce").fillna(0.0)
    if DUST_FLAG not in work.columns:
        work[DUST_FLAG] = work[CONTRIBUTION] > 0
    work[DUST_FLAG] = work[DUST_FLAG].fillna(False).astype(bool)
    if "station_class" not in work.columns:
        work["station_class"] = "unknown"
    work["station_class"] = work["station_class"].fillna("unknown").astype(str)
    return work.dropna(subset=["day", MEASURED])


def station_compliance(
    daily: pd.DataFrame, *, daily_limit: float, annual_limit: float, allowed_exceedances: int,
) -> pd.DataFrame:
    """One row per station: exceedances and annual mean before/after, and compliance status."""
    work = _prepare(daily)
    work["_exc_meas"] = work[MEASURED] > daily_limit
    work["_exc_adj"] = work[ADJUSTED] > daily_limit
    work["_exc_nat"] = work["_exc_meas"] & ~work["_exc_adj"]
    agg = {
        "station_class": ("station_class", "first"),
        "n_days": (MEASURED, "count"),
        "dust_days": (DUST_FLAG, "sum"),
        "exceedances_measured": ("_exc_meas", "sum"),
        "exceedances_adjusted": ("_exc_adj", "sum"),
        "exceedances_natural": ("_exc_nat", "sum"),
        "annual_mean_measured": (MEASURED, "mean"),
        "annual_mean_adjusted": (ADJUSTED, "mean"),
    }
    for col in ("Latitude", "Longitude"):
        if col in work.columns:
            agg[col] = (col, "first")
    table = work.groupby(STATION, as_index=False, observed=True).agg(**agg)
    table["annual_mean_reduction"] = table["annual_mean_measured"] - table["annual_mean_adjusted"]
    table["daily_lv_exceeded_measured"] = table["exceedances_measured"] > allowed_exceedances
    table["daily_lv_exceeded_adjusted"] = table["exceedances_adjusted"] > allowed_exceedances
    table["annual_lv_exceeded_measured"] = table["annual_mean_measured"] > annual_limit
    table["annual_lv_exceeded_adjusted"] = table["annual_mean_adjusted"] > annual_limit
    table["status_change"] = np.select(
        [
            table["daily_lv_exceeded_measured"] & ~table["daily_lv_exceeded_adjusted"],
            table["annual_lv_exceeded_measured"] & ~table["annual_lv_exceeded_adjusted"],
        ],
        ["daily LV met after deduction", "annual LV met after deduction"],
        default="",
    )
    return table.sort_values("exceedances_measured", ascending=False, ignore_index=True)


def compliance_summary(stations: pd.DataFrame) -> pd.DataFrame:
    """Headline numbers for the selected area, before and after the deduction."""
    rows = [
        ("Stations", len(stations), len(stations)),
        ("Stations with at least one exceedance day",
         int((stations["exceedances_measured"] > 0).sum()), int((stations["exceedances_adjusted"] > 0).sum())),
        ("Stations above the allowed number of exceedances (daily LV)",
         int(stations["daily_lv_exceeded_measured"].sum()), int(stations["daily_lv_exceeded_adjusted"].sum())),
        ("Stations above the annual LV",
         int(stations["annual_lv_exceeded_measured"].sum()), int(stations["annual_lv_exceeded_adjusted"].sum())),
        ("Exceedance days, all stations",
         int(stations["exceedances_measured"].sum()), int(stations["exceedances_adjusted"].sum())),
        ("Maximum exceedance days at one station",
         int(stations["exceedances_measured"].max()) if len(stations) else 0,
         int(stations["exceedances_adjusted"].max()) if len(stations) else 0),
        ("Mean annual concentration (µg/m³)",
         round(float(stations["annual_mean_measured"].mean()), 2),
         round(float(stations["annual_mean_adjusted"].mean()), 2)),
    ]
    table = pd.DataFrame(rows, columns=["indicator", "measured", "after deduction"])
    table["change"] = table["after deduction"] - table["measured"]
    return table


def class_summary(stations: pd.DataFrame) -> pd.DataFrame:
    """Exceedances and compliance per station class (RB, UB, TH)."""
    table = stations.groupby("station_class", observed=True).agg(
        stations=(STATION, "count"),
        exceedances_measured=("exceedances_measured", "sum"),
        exceedances_adjusted=("exceedances_adjusted", "sum"),
        exceedances_natural=("exceedances_natural", "sum"),
        daily_lv_exceeded_measured=("daily_lv_exceeded_measured", "sum"),
        daily_lv_exceeded_adjusted=("daily_lv_exceeded_adjusted", "sum"),
        mean_annual_reduction=("annual_mean_reduction", "mean"),
    )
    total = table["exceedances_measured"].replace(0, np.nan)
    table["share_natural_pct"] = (100 * table["exceedances_natural"] / total).round(1)
    return table.reset_index()


def monthly_summary(daily: pd.DataFrame, *, daily_limit: float) -> pd.DataFrame:
    """Per month over all stations: exceedance days, dust days and mean natural load."""
    work = _prepare(daily)
    work["month"] = work["day"].dt.month
    work["_exc_meas"] = work[MEASURED] > daily_limit
    work["_exc_adj"] = work[ADJUSTED] > daily_limit
    work["_contrib_dust"] = work[CONTRIBUTION].where(work[DUST_FLAG])
    table = work.groupby("month").agg(
        exceedances_measured=("_exc_meas", "sum"),
        exceedances_adjusted=("_exc_adj", "sum"),
        dust_station_days=(DUST_FLAG, "sum"),
        mean_contribution_on_dust_days=("_contrib_dust", "mean"),
    ).reindex(range(1, 13), fill_value=0)
    table["exceedances_natural"] = table["exceedances_measured"] - table["exceedances_adjusted"]
    table.index.name = "month"
    return table.reset_index()


def episode_table(daily: pd.DataFrame, *, daily_limit: float, top: int = 15) -> pd.DataFrame:
    """Days on which the most station exceedances are attributed to natural dust."""
    work = _prepare(daily)
    work["_exc_meas"] = work[MEASURED] > daily_limit
    work["_exc_nat"] = work["_exc_meas"] & ~(work[ADJUSTED] > daily_limit)
    table = work.groupby("day").agg(
        stations_with_dust=(DUST_FLAG, "sum"),
        exceedances_measured=("_exc_meas", "sum"),
        exceedances_natural=("_exc_nat", "sum"),
        max_contribution=(CONTRIBUTION, "max"),
        mean_contribution=(CONTRIBUTION, "mean"),
    )
    table = table[table["exceedances_natural"] > 0]
    return table.sort_values(["exceedances_natural", "max_contribution"], ascending=False).head(top).reset_index()


def export_tables(tables: dict[str, pd.DataFrame], out_dir: str, prefix: str) -> list[str]:
    """Write each table as ``{prefix}_{name}.csv``; returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, table in tables.items():
        path = os.path.join(out_dir, f"{prefix}_{name}.csv")
        table.to_csv(path, index=False)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_station_exceedances(stations: pd.DataFrame, *, allowed_exceedances: int, top: int = 30, ax=None):
    """Exceedance days per station, measured vs after deduction, with the allowed number."""
    import matplotlib.pyplot as plt

    view = stations[stations["exceedances_measured"] > 0].head(top).iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(view) + 1)))
    y = np.arange(len(view))
    ax.barh(y + 0.2, view["exceedances_measured"], height=0.4, color=MEASURED_COLOR, label="Measured")
    ax.barh(y - 0.2, view["exceedances_adjusted"], height=0.4, color=ADJUSTED_COLOR, label="After deduction")
    ax.axvline(allowed_exceedances, color="black", linestyle="--", label=f"Allowed ({allowed_exceedances} days)")
    labels = [f"{s} [{c}]" for s, c in zip(view[STATION], view["station_class"])]
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("Exceedance days")
    ax.set_title(f"Exceedance days per station (top {len(view)} by measured exceedances)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_monthly(monthly: pd.DataFrame, *, ax=None):
    """Monthly exceedance days (measured split into remaining and natural) and mean natural load."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    months = monthly["month"]
    ax.bar(months, monthly["exceedances_adjusted"], color=ADJUSTED_COLOR, label="Remaining after deduction")
    ax.bar(months, monthly["exceedances_natural"], bottom=monthly["exceedances_adjusted"],
           color=NATURAL_COLOR, label="Attributed to natural dust")
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("Exceedance days (all stations)")
    ax2 = ax.twinx()
    ax2.plot(months, monthly["mean_contribution_on_dust_days"], color="black", marker="o",
             label="Mean natural contribution on dust days")
    ax2.set_ylabel("µg/m³")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_title("Exceedances per month and natural contribution")
    return ax


def plot_annual_means(stations: pd.DataFrame, *, annual_limit: float, ax=None):
    """Annual mean measured vs after deduction; points crossing the limit change status."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for cls, marker in (("RB", "o"), ("UB", "s"), ("TH", "^"), ("unknown", "x")):
        rows = stations[stations["station_class"] == cls]
        if not rows.empty:
            ax.scatter(rows["annual_mean_measured"], rows["annual_mean_adjusted"], marker=marker, label=cls, alpha=0.8)
    high = float(np.nanmax([stations["annual_mean_measured"].max(), annual_limit])) * 1.1 if len(stations) else 1.0
    ax.plot([0, high], [0, high], color="grey", linewidth=0.8, label="1:1")
    ax.axvline(annual_limit, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(annual_limit, color="black", linestyle="--", linewidth=0.8, label=f"Annual LV ({annual_limit:g})")
    ax.set_xlim(0, high)
    ax.set_ylim(0, high)
    ax.set_xlabel("Annual mean measured (µg/m³)")
    ax.set_ylabel("Annual mean after deduction (µg/m³)")
    ax.set_title("Annual mean before and after deduction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_contribution_distribution(daily: pd.DataFrame, *, ax=None):
    """Natural contribution on dust days, per month."""
    import matplotlib.pyplot as plt

    work = _prepare(daily)
    work = work[work[DUST_FLAG] & (work[CONTRIBUTION] > 0)]
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    data = [work.loc[work["day"].dt.month == m, CONTRIBUTION].to_numpy() for m in range(1, 13)]
    ax.boxplot(data, showfliers=True, flierprops={"markersize": 2})
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("Natural contribution (µg/m³)")
    ax.set_title("Natural contribution on confirmed dust days, per month")
    ax.grid(axis="y", alpha=0.3)
    return ax
