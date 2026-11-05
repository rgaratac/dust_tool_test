"""
Sensitivity of the hourly dust deduction to the CAMS dust threshold.

The notebook flags a dust day when CAMS dust is above 5 µg/m³. This script
recomputes that deduction on the saved *hourly* Parquet for thresholds
3, 4, 5, 6, 7, 8, 9 and 10 µg/m³ and writes comparison plots for exceedance
days (measured daily mean above the pollutant limit).

The hourly product uses UTC days: EEA hourly stamps are treated as fixed
UTC+1 and then converted to UTC. This script does not read the daily product.

The saved Parquet is the target year. Neighbour medians are therefore taken
inside that year. The notebook also uses the days from 14 December of the
previous year, so the first days of January can differ slightly from a full
notebook run.

Run:
    python dust_threshold_sensitivity.py --parquet CAMS_dust_PM10_deduction_E2a_hour_2024_MA6.parquet
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cams_data_to_eea_daily as cams

# Inclusive integer thresholds, µg/m³. 5 is the notebook default.
THRESHOLDS = list(range(3, 11))
REFERENCE_THRESHOLD = 5.0
POLLUTANT = "PM10"
NEIGHBOR_N = 6
OUTPUT_DIR = "dust_threshold_sensitivity"


def _as_naive_utc_day(series: pd.Series) -> pd.Series:
    """UTC calendar day as timezone-naive timestamps, for a stable join key."""
    stamps = pd.to_datetime(series, utc=True, errors="coerce")
    return stamps.dt.tz_localize(None).dt.floor("D")


def apply_dust_threshold(
    df: pd.DataFrame,
    threshold: float,
    *,
    pollutant: str,
    neighbor_n: int,
    time_col: str = "day",
    value_col: str = "daily_mean",
) -> pd.DataFrame:
    """Rebuild dust flags, the neighbour median, and the corrected concentration.

    ``Exceedance`` stays the measured daily mean above the pollutant limit.
    The background median is filled only on days that are both a dust day and
    an exceedance, matching the notebook. A dust day is ``cams_dust`` strictly
    above ``threshold``.
    """
    work = df.copy()
    work[time_col] = _as_naive_utc_day(work[time_col])
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").astype("float32")
    work["cams_dust"] = pd.to_numeric(work["cams_dust"], errors="coerce").astype("float32")
    limit = cams.pollutant_daily_threshold(pollutant)
    work["dust_flag"] = work["cams_dust"] > float(threshold)
    work["Exceedance"] = work[value_col] > limit
    work["dust_threshold"] = float(threshold)
    work["pollutant_median"] = np.float32(np.nan)

    for station, st in work.groupby("Samplingpoint", sort=False, observed=False):
        medians = cams.compute_station_baseline(
            st,
            st,
            time_col=time_col,
            value_col=value_col,
            neighbor_n=neighbor_n,
        )
        work.loc[st.index, "pollutant_median"] = medians.to_numpy(dtype="float32")

    dust_mask = work["dust_flag"].fillna(False).astype(bool)
    work["Dust_contribution"] = np.where(
        dust_mask,
        work[value_col] - work["pollutant_median"],
        0.0,
    ).astype("float32")
    work["corrected_pollutant"] = np.where(
        dust_mask,
        work["pollutant_median"],
        work[value_col],
    ).astype("float32")
    work["Dust_contribution"] = work["Dust_contribution"].clip(lower=0)
    work["Dust_exceedance"] = (work[value_col] > limit) & (work["corrected_pollutant"] <= limit)
    return work


def threshold_sweep(
    df: pd.DataFrame,
    thresholds: list[float],
    *,
    pollutant: str,
    neighbor_n: int,
) -> dict[float, pd.DataFrame]:
    """One recomputed hourly table per dust threshold."""
    return {
        float(threshold): apply_dust_threshold(
            df,
            float(threshold),
            pollutant=pollutant,
            neighbor_n=neighbor_n,
        )
        for threshold in thresholds
    }


def exceedance_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Days whose measured concentration is above the pollutant limit."""
    return df.loc[df["Exceedance"].fillna(False).astype(bool)].copy()


def _row_key(df: pd.DataFrame) -> pd.Series:
    day = _as_naive_utc_day(df["day"]).dt.strftime("%Y-%m-%d")
    return df["Samplingpoint"].astype(str) + "|" + day


def exceedance_summary(frames: dict[float, pd.DataFrame]) -> pd.DataFrame:
    """One row per threshold: how exceedance days respond to the dust cut."""
    rows = []
    for threshold, frame in frames.items():
        exc = exceedance_rows(frame)
        rows.append(
            {
                "dust_threshold_ug_m3": threshold,
                "exceedance_days": int(len(exc)),
                "dust_days": int(frame["dust_flag"].fillna(False).astype(bool).sum()),
                "dust_exceedance_days": int(exc["Dust_exceedance"].fillna(False).astype(bool).sum()),
                "median_dust_contribution_on_exceedance": float(exc["Dust_contribution"].median())
                if len(exc)
                else np.nan,
                "median_corrected_on_exceedance": float(exc["corrected_pollutant"].median())
                if len(exc)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("dust_threshold_ug_m3").reset_index(drop=True)


def _reference(frames: dict[float, pd.DataFrame], reference: float) -> float:
    if float(reference) in frames:
        return float(reference)
    return float(min(frames, key=lambda value: abs(value - float(reference))))


def plot_contribution_scatter(
    frames: dict[float, pd.DataFrame],
    path: str,
    *,
    reference: float = REFERENCE_THRESHOLD,
) -> str:
    """Exceedance-day dust contribution at each threshold against the reference."""
    ref_key = _reference(frames, reference)
    others = [key for key in sorted(frames) if key != ref_key]
    if not others:
        raise ValueError("Need at least one threshold besides the reference.")
    ref = exceedance_rows(frames[ref_key])
    ref = ref.assign(_key=_row_key(ref))
    ncols = 4
    nrows = int(np.ceil(len(others) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax, key in zip(flat, others):
        other = exceedance_rows(frames[key])
        other = other.assign(_key=_row_key(other))
        paired = ref[["_key", "Dust_contribution"]].merge(
            other[["_key", "Dust_contribution"]],
            on="_key",
            how="inner",
            suffixes=("_ref", "_other"),
        )
        ax.scatter(
            paired["Dust_contribution_ref"],
            paired["Dust_contribution_other"],
            s=12,
            alpha=0.55,
            c="#1f4e79",
            linewidths=0,
        )
        if len(paired):
            hi = float(
                np.nanmax(
                    [
                        paired["Dust_contribution_ref"].max(),
                        paired["Dust_contribution_other"].max(),
                        1.0,
                    ]
                )
            )
            ax.plot([0, hi], [0, hi], color="#b0b0b0", linewidth=1)
        ax.set_title(f"{key:g} µg/m³")
        ax.set_xlabel(f"Contribution at {ref_key:g} µg/m³")
        ax.set_ylabel(f"Contribution at {key:g} µg/m³")
        ax.grid(True, alpha=0.3)
    for ax in flat[len(others) :]:
        ax.axis("off")
    fig.suptitle(
        "Hourly data, exceedance days: dust contribution vs the reference threshold\n"
        "Each point is one station-day. The day is a UTC day.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dust_exceedance_scatter(
    frames: dict[float, pd.DataFrame],
    path: str,
    *,
    reference: float = REFERENCE_THRESHOLD,
) -> str:
    """Station counts of exceedance days removed as dust, against the reference."""
    ref_key = _reference(frames, reference)

    def counts(frame: pd.DataFrame) -> pd.Series:
        exc = exceedance_rows(frame)
        flagged = exc.loc[exc["Dust_exceedance"].fillna(False).astype(bool)]
        return flagged.groupby(flagged["Samplingpoint"].astype(str)).size()

    ref_counts = counts(frames[ref_key])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(frames) - 1, 1)))
    color_i = 0
    hi = float(ref_counts.max()) if len(ref_counts) else 1.0
    for key in sorted(frames):
        if key == ref_key:
            continue
        other = counts(frames[key])
        both = pd.concat(
            [ref_counts.rename("ref"), other.rename("other")],
            axis=1,
        ).fillna(0.0)
        hi = max(hi, float(both.to_numpy().max()) if len(both) else 1.0)
        ax.scatter(
            both["ref"],
            both["other"],
            s=28,
            alpha=0.75,
            color=colors[color_i],
            label=f"{key:g} µg/m³",
        )
        color_i += 1
    ax.plot([0, hi], [0, hi], color="#b0b0b0", linewidth=1)
    ax.set_xlabel(f"Dust-exceedance days at {ref_key:g} µg/m³")
    ax.set_ylabel("Dust-exceedance days at the other threshold")
    ax.set_title(
        "Hourly data: exceedance days attributed to dust, per station\n"
        "Compared with the reference threshold. Days are UTC days."
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="Threshold", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_contribution_distributions(frames: dict[float, pd.DataFrame], path: str) -> str:
    """Boxplots of the dust contribution on exceedance days."""
    labels = []
    series = []
    for key in sorted(frames):
        exc = exceedance_rows(frames[key])
        labels.append(f"{key:g}")
        series.append(exc["Dust_contribution"].dropna().to_numpy())
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.boxplot(series, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_xlabel("CAMS dust threshold (µg/m³)")
    ax.set_ylabel("Dust contribution on exceedance days (µg/m³)")
    ax.set_title(
        "Hourly data: distribution of the dust contribution on exceedance days\n"
        "Outliers are hidden. The day is a UTC day."
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_corrected_distributions(frames: dict[float, pd.DataFrame], path: str) -> str:
    """Overlaid histograms of the corrected concentration on exceedance days."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(frames)))
    for color, key in zip(colors, sorted(frames)):
        values = exceedance_rows(frames[key])["corrected_pollutant"].dropna()
        if values.empty:
            continue
        ax.hist(
            values,
            bins=20,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=f"{key:g} µg/m³",
        )
    ax.set_xlabel("Corrected concentration on exceedance days (µg/m³)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Hourly data: corrected concentration on exceedance days\n"
        "Each line is one CAMS dust threshold. The day is a UTC day."
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="Threshold", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def write_sensitivity_plots(
    frames: dict[float, pd.DataFrame],
    output_dir: str,
    *,
    reference: float = REFERENCE_THRESHOLD,
) -> dict[str, str]:
    """Write the comparison figures and return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "scatter_contribution": os.path.join(output_dir, "scatter_contribution_exceedance.png"),
        "scatter_dust_exceedance_days": os.path.join(
            output_dir, "scatter_dust_exceedance_days.png"
        ),
        "distribution_contribution": os.path.join(
            output_dir, "distribution_contribution_exceedance.png"
        ),
        "distribution_corrected": os.path.join(output_dir, "distribution_corrected_exceedance.png"),
    }
    plot_contribution_scatter(frames, paths["scatter_contribution"], reference=reference)
    plot_dust_exceedance_scatter(frames, paths["scatter_dust_exceedance_days"], reference=reference)
    plot_contribution_distributions(frames, paths["distribution_contribution"])
    plot_corrected_distributions(frames, paths["distribution_corrected"])
    return paths


def run_sensitivity(
    parquet_path: str,
    output_dir: str,
    *,
    pollutant: str = POLLUTANT,
    neighbor_n: int = NEIGHBOR_N,
    thresholds: list[float] | None = None,
    reference: float = REFERENCE_THRESHOLD,
) -> pd.DataFrame:
    """Load an hourly deduction Parquet, sweep the thresholds, and save plots."""
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(
            f"Hourly deduction Parquet not found: {parquet_path}. "
            "Point --parquet at CAMS_dust_{{pollutant}}_deduction_{{dataset}}_hour_{{year}}_MA{{n}}.parquet."
        )
    thresholds = list(THRESHOLDS if thresholds is None else thresholds)
    df = pd.read_parquet(parquet_path)
    needed = {"Samplingpoint", "day", "daily_mean", "cams_dust"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"{parquet_path} is missing columns {sorted(missing)}")
    frames = threshold_sweep(df, thresholds, pollutant=pollutant, neighbor_n=neighbor_n)
    summary = exceedance_summary(frames)
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "threshold_summary.csv")
    summary.to_csv(summary_path, index=False)
    paths = write_sensitivity_plots(frames, output_dir, reference=reference)
    print(f"Wrote {summary_path}")
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    return summary


# ---------------------------------------------------------------------------
# Model-specific threshold with the paired natural-contribution methods
# ---------------------------------------------------------------------------

def natural_threshold_sweep(
    df: pd.DataFrame,
    pairing: pd.DataFrame,
    thresholds: list[float],
    *,
    params=None,
    daily_limit: float | None = None,
) -> dict[float, pd.DataFrame]:
    """One natural-contribution table per CAMS dust threshold (event days change with it)."""
    import natural_contribution as nc

    params = params or nc.MethodParams()
    return {
        float(t): nc.compute_natural_contribution(
            df, pairing, dust_threshold=float(t), params=params, daily_limit=daily_limit
        )
        for t in sorted(thresholds)
    }


def contribution_consistency(
    frames: dict[float, pd.DataFrame],
    *,
    group_col: str = "country",
    station_col: str = "Samplingpoint",
) -> pd.DataFrame:
    """Per group and threshold: CAMS dust days and the share with a contribution > 0.

    ``group_col='country'`` is derived from the first two characters of the
    station id when the column is missing. Any other column (e.g. a region)
    must already be in the tables.
    """
    rows = []
    for threshold, frame in frames.items():
        work = frame.loc[frame["cams_dust_flag"] & ~frame["insufficient_reference"]].copy()
        if group_col not in work.columns and group_col == "country":
            work["country"] = work[station_col].astype(str).str[:2]
        for group, sub in work.groupby(group_col, observed=True):
            confirmed = sub["natural_contribution"] > 0
            rows.append({
                group_col: group,
                "dust_threshold_ug_m3": threshold,
                "cams_dust_days": int(len(sub)),
                "confirmed_days": int(confirmed.sum()),
                "share_confirmed": float(confirmed.mean()) if len(sub) else np.nan,
                "median_contribution": float(sub["natural_contribution"].median()) if len(sub) else np.nan,
            })
    return pd.DataFrame(rows).sort_values([group_col, "dust_threshold_ug_m3"]).reset_index(drop=True)


def lowest_consistent_threshold(
    consistency: pd.DataFrame,
    *,
    group_col: str = "country",
    min_share: float = 0.9,
    min_days: int = 10,
) -> pd.DataFrame:
    """Lowest threshold from which every higher threshold has ``share_confirmed >= min_share``.

    Thresholds with fewer than ``min_days`` CAMS dust days are ignored. The
    result is NaN for a group where no threshold qualifies.
    """
    rows = []
    for group, sub in consistency.groupby(group_col):
        sub = sub[sub["cams_dust_days"] >= min_days].sort_values("dust_threshold_ug_m3")
        best = np.nan
        ok = (sub["share_confirmed"] >= min_share).to_numpy()
        for i in range(len(sub)):
            if ok[i:].all():
                best = float(sub["dust_threshold_ug_m3"].iloc[i])
                break
        share = sub.loc[sub["dust_threshold_ug_m3"] == best, "share_confirmed"]
        rows.append({
            group_col: group,
            "model_threshold_ug_m3": best,
            "share_confirmed_at_threshold": float(share.iloc[0]) if len(share) else np.nan,
            "thresholds_tested": int(len(sub)),
        })
    return pd.DataFrame(rows)


def _parse_thresholds(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one threshold.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        required=True,
        help="Hourly deduction Parquet (EEA_temporal_flag hour).",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--pollutant", default=POLLUTANT, choices=["PM10", "PM2.5"])
    parser.add_argument("--neighbor-n", type=int, default=NEIGHBOR_N)
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=THRESHOLDS,
        help="Comma-separated µg/m³ values. Default: 3,4,5,6,7,8,9,10.",
    )
    parser.add_argument("--reference", type=float, default=REFERENCE_THRESHOLD)
    args = parser.parse_args()
    run_sensitivity(
        args.parquet,
        args.output_dir,
        pollutant=args.pollutant,
        neighbor_n=args.neighbor_n,
        thresholds=args.thresholds,
        reference=args.reference,
    )


if __name__ == "__main__":
    main()
