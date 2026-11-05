"""
Station classification and station pairing for the natural-contribution methods.

Every function here is pure pandas/numpy, so it can be tested without Colab,
widgets or map libraries. The interactive map and the editor live in
``pairing_viz.py``.

Classes
-------
RB  regional background: rural* area and background type
UB  urban or suburban background
TH  traffic or industrial (hotspot)

Methods
-------
option1            the station uses the net load of its paired RB station
option2            UB self-calibration on its own measurements
option2_ub_paired  TH station that uses the Option 2 load of its paired UB
none               no valid reference, nothing is subtracted

The pairing table is the single source the deduction reads. It can be edited
with ``set_manual_pair`` / ``reset_station`` or round-tripped through CSV with
``export_pairing_csv`` and ``apply_user_pairing``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STATION_COLUMNS = [
    "station_id", "station_name", "country", "class", "area", "type",
    "latitude", "longitude", "altitude",
]
PAIRING_COLUMNS = STATION_COLUMNS + [
    "paired_rb_id", "paired_rb_name", "rb_distance_km", "rb_altitude_diff_m",
    "paired_ub_id", "paired_ub_name", "ub_distance_km",
    "method", "source", "comment",
]
EDITABLE_COLUMNS = ["paired_rb_id", "paired_ub_id", "method", "comment"]
METHODS = ("option1", "option2", "option2_ub_paired", "none")
CLASSES = ("RB", "UB", "TH")
USER_PAIRING_FILENAME = "station_pairing_user.csv"
DEFAULT_PAIRING_FILENAME = "station_pairing_default.csv"

_KEEP = object()


@dataclass(frozen=True)
class PairingLimits:
    """Distance and altitude limits for a suitable reference station."""

    max_rb_distance_km: float = 100.0
    max_rb_altitude_diff_m: float | None = 500.0
    max_ub_distance_km: float = 50.0
    max_ub_altitude_diff_m: float | None = None
    same_country: bool = True


class PairingValidationError(ValueError):
    """Raised when an edited pairing refers to unknown or wrong-class stations."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("Invalid station pairing:\n- " + "\n- ".join(self.errors))


@dataclass
class PairingImportReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    changes: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["station_id", "field", "default_value", "new_value"]
        )
    )

    def summary(self) -> str:
        n_stations = self.changes["station_id"].nunique() if len(self.changes) else 0
        lines = [f"{n_stations} station(s) differ from the default pairing."]
        lines += [f"ERROR: {msg}" for msg in self.errors]
        lines += [f"Warning: {msg}" for msg in self.warnings]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classification and station table
# ---------------------------------------------------------------------------

def _text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower()


def classify_station(area, station_type) -> str:
    """RB, UB, TH, or ``unknown`` from the EEA area and type labels."""
    a, t = _text(area), _text(station_type)
    if t in ("traffic", "industrial"):
        return "TH"
    if t == "background":
        if a.startswith("rural"):
            return "RB"
        if a in ("urban", "suburban"):
            return "UB"
    return "unknown"


def samplingpoint_key(metadata: pd.DataFrame) -> pd.Series:
    """Join key used by the notebook: EoI country prefix + '/' + sampling point id."""
    return (
        metadata["Air Quality Station EoI Code"].astype(str).str[:2]
        + "/"
        + metadata["Sampling Point Id"].astype(str)
    )


def build_station_table(
    metadata: pd.DataFrame,
    *,
    pollutant: str | None = None,
    station_ids=None,
    countries=None,
) -> pd.DataFrame:
    """One row per sampling point with the columns in ``STATION_COLUMNS``.

    ``station_ids`` restricts the table to the stations that have data, so a
    station is never paired with a reference that has no measurements.
    """
    meta = metadata.copy()
    if pollutant and "Air Pollutant" in meta.columns:
        subset = meta[meta["Air Pollutant"] == pollutant]
        if not subset.empty:
            meta = subset
    meta["station_id"] = samplingpoint_key(meta)
    meta = meta.drop_duplicates(subset=["station_id"], keep="first")
    out = pd.DataFrame(
        {
            "station_id": meta["station_id"].astype(str),
            "station_name": meta.get("Air Quality Station Name", meta["station_id"]).astype(str),
            "country": meta["station_id"].str[:2],
            "area": meta.get("Air Quality Station Area", pd.Series("", index=meta.index)).astype(str),
            "type": meta.get("Air Quality Station Type", pd.Series("", index=meta.index)).astype(str),
            "latitude": pd.to_numeric(meta["Latitude"], errors="coerce"),
            "longitude": pd.to_numeric(meta["Longitude"], errors="coerce"),
            "altitude": pd.to_numeric(meta.get("Altitude"), errors="coerce"),
        }
    )
    out["class"] = [classify_station(a, t) for a, t in zip(out["area"], out["type"])]
    if station_ids is not None:
        out = out[out["station_id"].isin(set(map(str, station_ids)))]
    if countries is not None:
        out = out[out["country"].isin(set(countries))]
    out = out.dropna(subset=["latitude", "longitude"])
    return out[STATION_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Inputs broadcast like numpy arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _within(dist, alt_diff, max_km, max_alt):
    ok = dist <= max_km
    if max_alt is not None:
        # Unknown altitude cannot be checked, so it does not block a pair.
        ok &= np.isnan(alt_diff) | (alt_diff <= max_alt)
    return ok


def _nearest_reference(
    src: pd.DataFrame,
    cand: pd.DataFrame,
    max_km: float,
    max_alt: float | None,
    same_country: bool,
) -> pd.DataFrame:
    """Nearest suitable candidate for each source row (NaN when none)."""
    result = pd.DataFrame(
        {"ref_id": None, "distance_km": np.nan, "altitude_diff_m": np.nan},
        index=src.index,
    ).astype({"ref_id": object})
    if src.empty or cand.empty:
        return result
    groups = src.groupby("country").groups if same_country else {None: src.index}
    for country, idx in groups.items():
        s = src.loc[idx]
        c = cand if country is None else cand[cand["country"] == country]
        if c.empty:
            continue
        dist = haversine_km(
            s["latitude"].to_numpy()[:, None], s["longitude"].to_numpy()[:, None],
            c["latitude"].to_numpy()[None, :], c["longitude"].to_numpy()[None, :],
        )
        alt = np.abs(s["altitude"].to_numpy(dtype=float)[:, None] - c["altitude"].to_numpy(dtype=float)[None, :])
        ok = _within(dist, alt, max_km, max_alt)
        ok &= s["station_id"].to_numpy()[:, None] != c["station_id"].to_numpy()[None, :]
        masked = np.where(ok, dist, np.inf)
        best = masked.argmin(axis=1)
        rows = np.arange(len(s))
        found = np.isfinite(masked[rows, best])
        result.loc[s.index[found], "ref_id"] = c["station_id"].to_numpy()[best[found]]
        result.loc[s.index[found], "distance_km"] = dist[rows, best][found]
        result.loc[s.index[found], "altitude_diff_m"] = alt[rows, best][found]
    return result


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def _clean_id(value):
    text = "" if value is None else str(value).strip()
    if text.lower() in ("", "nan", "none", "<na>", "nat"):
        return None
    return text


def _clean_method(value):
    text = "" if value is None else str(value).strip().lower()
    return None if text in ("", "nan", "<na>") else text


def _clean_ids(series: pd.Series) -> pd.Series:
    return pd.Series([_clean_id(v) for v in series], index=series.index, dtype=object)


def derive_method(station_class: str, paired_rb_id, paired_ub_id) -> str:
    """Method implied by the pairing alone."""
    if _clean_id(paired_rb_id):
        return "option1"
    if station_class in ("UB", "RB"):
        return "option2"
    if station_class == "TH" and _clean_id(paired_ub_id):
        return "option2_ub_paired"
    return "none"


def resolve_method(station_class: str, paired_rb_id, paired_ub_id, requested=None):
    """Requested method if it is consistent with the pairing, else the derived one.

    Returns ``(method, warning)``. ``warning`` is None when the request is used.
    """
    derived = derive_method(station_class, paired_rb_id, paired_ub_id)
    req = _clean_method(requested)
    if req is None:
        return derived, None
    if req not in METHODS:
        return derived, f"unknown method {requested!r}; using {derived}"
    if req == "none":
        return "none", None
    if req == "option1" and not _clean_id(paired_rb_id):
        return derived, f"option1 needs a paired RB; using {derived}"
    if req == "option2" and station_class not in ("UB", "RB"):
        return derived, f"option2 self-calibration is only for UB/RB stations; using {derived}"
    if req == "option2_ub_paired" and not (station_class == "TH" and _clean_id(paired_ub_id)):
        return derived, f"option2_ub_paired needs a TH station with a paired UB; using {derived}"
    return req, None


# ---------------------------------------------------------------------------
# Default pairing
# ---------------------------------------------------------------------------

def default_pairing(stations: pd.DataFrame, limits: PairingLimits = PairingLimits()) -> pd.DataFrame:
    """Nearest suitable RB for every station; a TH without RB gets the nearest UB.

    Each station has at most one reference. An RB station is its own reference
    (distance 0), so its Option 1 load is also subtracted at the RB itself.
    """
    st = stations.reset_index(drop=True).copy()
    rb = st[st["class"] == "RB"]
    ub = st[st["class"] == "UB"]

    rb_pair = _nearest_reference(
        st[st["class"] != "RB"], rb,
        limits.max_rb_distance_km, limits.max_rb_altitude_diff_m, limits.same_country,
    )
    has_rb = rb_pair.index[rb_pair["ref_id"].notna()]
    th_without_rb = st[(st["class"] == "TH") & ~st.index.isin(has_rb)]
    ub_pair = _nearest_reference(
        th_without_rb, ub,
        limits.max_ub_distance_km, limits.max_ub_altitude_diff_m, limits.same_country,
    )

    table = st.copy()
    table["paired_rb_id"] = None
    table["rb_distance_km"] = np.nan
    table["rb_altitude_diff_m"] = np.nan
    table.loc[rb_pair.index, "paired_rb_id"] = rb_pair["ref_id"]
    table.loc[rb_pair.index, "rb_distance_km"] = rb_pair["distance_km"]
    table.loc[rb_pair.index, "rb_altitude_diff_m"] = rb_pair["altitude_diff_m"]
    is_rb = table["class"] == "RB"
    table.loc[is_rb, "paired_rb_id"] = table.loc[is_rb, "station_id"]
    table.loc[is_rb, ["rb_distance_km", "rb_altitude_diff_m"]] = 0.0

    table["paired_ub_id"] = None
    table["ub_distance_km"] = np.nan
    table.loc[ub_pair.index, "paired_ub_id"] = ub_pair["ref_id"]
    table.loc[ub_pair.index, "ub_distance_km"] = ub_pair["distance_km"]

    table["method"] = [
        derive_method(c, r, u)
        for c, r, u in zip(table["class"], table["paired_rb_id"], table["paired_ub_id"])
    ]
    table["source"] = "default"
    table["comment"] = ""
    return fill_pairing_metrics(table, st)


def fill_pairing_metrics(table: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Refresh station attributes, reference names, distances and altitude differences."""
    st = stations.set_index("station_id")
    out = table.copy()
    out["station_id"] = out["station_id"].astype(str)
    for col in ("station_name", "country", "class", "area", "type", "latitude", "longitude", "altitude"):
        out[col] = out["station_id"].map(st[col])
    out["paired_rb_id"] = _clean_ids(out["paired_rb_id"])
    out["paired_ub_id"] = _clean_ids(out["paired_ub_id"])
    # One reference per station: an RB pair wins, and only TH stations use a UB pair.
    drop_ub = out["paired_rb_id"].notna() | (out["class"] != "TH")
    out.loc[drop_ub, "paired_ub_id"] = None
    out["paired_rb_name"] = out["paired_rb_id"].map(st["station_name"])
    out["paired_ub_name"] = out["paired_ub_id"].map(st["station_name"])

    def _metrics(ref_col):
        lat2 = out[ref_col].map(st["latitude"]).astype(float)
        lon2 = out[ref_col].map(st["longitude"]).astype(float)
        alt2 = out[ref_col].map(st["altitude"]).astype(float)
        dist = haversine_km(out["latitude"].astype(float), out["longitude"].astype(float), lat2, lon2)
        alt = np.abs(out["altitude"].astype(float) - alt2)
        return np.round(dist, 2), np.round(alt, 1)

    out["rb_distance_km"], out["rb_altitude_diff_m"] = _metrics("paired_rb_id")
    out["ub_distance_km"], _ = _metrics("paired_ub_id")
    out["comment"] = out["comment"].fillna("").astype(str) if "comment" in out else ""
    if "source" not in out:
        out["source"] = "default"
    return out[PAIRING_COLUMNS].reset_index(drop=True)


def pairing_limit_warnings(row, limits: PairingLimits) -> list[str]:
    """Human-readable warnings when a pair is outside the configured limits."""
    msgs = []
    sid = row["station_id"]
    if _clean_id(row.get("paired_rb_id")) and row["paired_rb_id"] != sid:
        d, a = row.get("rb_distance_km"), row.get("rb_altitude_diff_m")
        if pd.notna(d) and d > limits.max_rb_distance_km:
            msgs.append(f"{sid}: RB {row['paired_rb_id']} is {d:.1f} km away (max {limits.max_rb_distance_km:g} km)")
        if limits.max_rb_altitude_diff_m is not None and pd.notna(a) and a > limits.max_rb_altitude_diff_m:
            msgs.append(f"{sid}: RB {row['paired_rb_id']} altitude differs by {a:.0f} m (max {limits.max_rb_altitude_diff_m:g} m)")
        if limits.same_country and str(row["paired_rb_id"])[:2] != str(sid)[:2]:
            msgs.append(f"{sid}: RB {row['paired_rb_id']} is in another country")
    if _clean_id(row.get("paired_ub_id")):
        d = row.get("ub_distance_km")
        if pd.notna(d) and d > limits.max_ub_distance_km:
            msgs.append(f"{sid}: UB {row['paired_ub_id']} is {d:.1f} km away (max {limits.max_ub_distance_km:g} km)")
    return msgs


# ---------------------------------------------------------------------------
# Manual editing
# ---------------------------------------------------------------------------

def candidate_references(
    stations: pd.DataFrame,
    station_id: str,
    ref_class: str,
    *,
    limits: PairingLimits = PairingLimits(),
    max_candidates: int | None = 50,
) -> pd.DataFrame:
    """Possible RB or UB references for one station, nearest first."""
    st = stations.set_index("station_id")
    if station_id not in st.index:
        raise KeyError(f"Unknown station {station_id!r}")
    me = st.loc[station_id]
    cand = stations[(stations["class"] == ref_class) & (stations["station_id"] != station_id)].copy()
    if limits.same_country:
        cand = cand[cand["country"] == me["country"]]
    cand["distance_km"] = haversine_km(me["latitude"], me["longitude"], cand["latitude"], cand["longitude"])
    cand["altitude_diff_m"] = np.abs(float(me["altitude"]) - cand["altitude"].astype(float))
    if ref_class == "RB":
        max_km, max_alt = limits.max_rb_distance_km, limits.max_rb_altitude_diff_m
    else:
        max_km, max_alt = limits.max_ub_distance_km, limits.max_ub_altitude_diff_m
    cand["within_limits"] = _within(
        cand["distance_km"].to_numpy(), cand["altitude_diff_m"].to_numpy(), max_km, max_alt
    )
    cand = cand.sort_values("distance_km")
    if max_candidates is not None:
        cand = cand.head(max_candidates)
    return cand[["station_id", "station_name", "distance_km", "altitude_diff_m", "within_limits"]].reset_index(drop=True)


def _check_reference(stations_idx: pd.DataFrame, ref_id, expected: str, label: str) -> list[str]:
    ref = _clean_id(ref_id)
    if ref is None:
        return []
    if ref not in stations_idx.index:
        return [f"{label}: unknown station id {ref!r}"]
    actual = stations_idx.at[ref, "class"]
    if actual != expected:
        return [f"{label}: {ref} is a {actual} station, not {expected}"]
    return []


def set_manual_pair(
    table: pd.DataFrame,
    stations: pd.DataFrame,
    station_id: str,
    *,
    rb_id=_KEEP,
    ub_id=_KEEP,
    method=None,
    comment=None,
    limits: PairingLimits = PairingLimits(),
):
    """Return ``(new_table, warnings)`` with one station edited and marked manual.

    ``rb_id=None`` removes the RB pair (forces Option 2 for UB, the UB pair for
    TH). ``method='none'`` excludes the station from the deduction.
    """
    st = stations.set_index("station_id")
    if station_id not in set(table["station_id"]):
        raise PairingValidationError([f"unknown station id {station_id!r}"])
    out = table.copy()
    i = out.index[out["station_id"] == station_id][0]
    new_rb = out.at[i, "paired_rb_id"] if rb_id is _KEEP else _clean_id(rb_id)
    new_ub = out.at[i, "paired_ub_id"] if ub_id is _KEEP else _clean_id(ub_id)
    errors = _check_reference(st, new_rb, "RB", f"{station_id} paired RB")
    errors += _check_reference(st, new_ub, "UB", f"{station_id} paired UB")
    if errors:
        raise PairingValidationError(errors)
    out.at[i, "paired_rb_id"] = new_rb
    out.at[i, "paired_ub_id"] = new_ub
    if comment is not None:
        out.at[i, "comment"] = str(comment)
    out.at[i, "source"] = "manual"
    out = fill_pairing_metrics(out, stations)
    i = out.index[out["station_id"] == station_id][0]
    resolved, warning = resolve_method(out.at[i, "class"], new_rb, new_ub, method)
    out.at[i, "method"] = resolved
    warnings = [f"{station_id}: {warning}"] if warning else []
    warnings += pairing_limit_warnings(out.loc[i], limits)
    return out, warnings


def reset_station(table: pd.DataFrame, default: pd.DataFrame, station_id: str) -> pd.DataFrame:
    """Put one station back to its default pairing."""
    out = table.copy()
    src = default[default["station_id"] == station_id]
    if src.empty:
        raise KeyError(f"Unknown station {station_id!r}")
    i = out.index[out["station_id"] == station_id]
    out.loc[i, PAIRING_COLUMNS] = src[PAIRING_COLUMNS].to_numpy()
    return out


def reset_all(default: pd.DataFrame) -> pd.DataFrame:
    return default.copy()


# ---------------------------------------------------------------------------
# CSV round trip
# ---------------------------------------------------------------------------

def export_pairing_csv(table: pd.DataFrame, path: str) -> str:
    table[PAIRING_COLUMNS].to_csv(path, index=False)
    return path


def read_pairing_csv(path: str) -> pd.DataFrame:
    """Read an edited pairing CSV; ids stay strings, blanks become None."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _diff_rows(default_row, new_row) -> list[tuple]:
    rows = []
    for col in ("paired_rb_id", "paired_ub_id", "method"):
        clean = _clean_method if col == "method" else _clean_id
        a, b = clean(default_row[col]), clean(new_row[col])
        if a != b:
            rows.append((new_row["station_id"], col, a, b))
    return rows


def apply_user_pairing(
    user: pd.DataFrame,
    stations: pd.DataFrame,
    default: pd.DataFrame,
    *,
    limits: PairingLimits = PairingLimits(),
    strict: bool = True,
):
    """Validate an edited pairing table and merge it onto the default.

    Only ``EDITABLE_COLUMNS`` are taken from ``user``; names, distances and
    altitude differences are recomputed. Wrong ids or wrong classes are errors
    (raised when ``strict``; otherwise those rows keep the default). Pairs
    outside the limits only produce warnings. Returns ``(table, report)``.
    """
    report = PairingImportReport()
    if "station_id" not in user.columns:
        report.errors.append("the CSV has no 'station_id' column")
        raise PairingValidationError(report.errors)
    st = stations.set_index("station_id")
    user = user.copy()
    for col in EDITABLE_COLUMNS:
        if col not in user.columns:
            user[col] = ""
    user["station_id"] = user["station_id"].map(_clean_id)

    dupes = user["station_id"][user["station_id"].duplicated() & user["station_id"].notna()]
    for sid in sorted(set(dupes)):
        report.errors.append(f"station {sid} appears more than once")

    table = default.copy().set_index("station_id")
    bad_rows = set(dupes)
    for _, row in user.iterrows():
        sid = row["station_id"]
        if sid is None:
            continue
        if sid not in table.index:
            report.errors.append(f"unknown station id {sid!r}")
            continue
        errs = _check_reference(st, row["paired_rb_id"], "RB", f"{sid} paired RB")
        errs += _check_reference(st, row["paired_ub_id"], "UB", f"{sid} paired UB")
        m = _clean_method(row["method"])
        if m is not None and m not in METHODS:
            errs.append(f"{sid}: method {m!r} is not one of {', '.join(METHODS)}")
        if errs:
            report.errors.extend(errs)
            bad_rows.add(sid)
            continue
        if sid in bad_rows:
            continue
        table.at[sid, "paired_rb_id"] = _clean_id(row["paired_rb_id"])
        table.at[sid, "paired_ub_id"] = _clean_id(row["paired_ub_id"])
        table.at[sid, "comment"] = "" if row["comment"] is None else str(row["comment"])
        table.at[sid, "method"] = m

    if report.errors and strict:
        raise PairingValidationError(report.errors)

    missing = set(default["station_id"]) - set(user["station_id"].dropna())
    if missing:
        report.warnings.append(f"{len(missing)} station(s) not in the CSV keep the default pairing")

    table = fill_pairing_metrics(table.reset_index(), stations)
    defaults = default.set_index("station_id")
    changes = []
    for i, row in table.iterrows():
        d_row = defaults.loc[row["station_id"]]
        pair_changed = (
            _clean_id(row["paired_rb_id"]) != _clean_id(d_row["paired_rb_id"])
            or _clean_id(row["paired_ub_id"]) != _clean_id(d_row["paired_ub_id"])
        )
        requested = row["method"]
        # A method left unchanged while the pair changed is re-derived.
        if pair_changed and _clean_method(requested) == d_row["method"]:
            requested = None
        method, warning = resolve_method(row["class"], row["paired_rb_id"], row["paired_ub_id"], requested)
        if warning:
            report.warnings.append(f"{row['station_id']}: {warning}")
        table.at[i, "method"] = method
        diff = _diff_rows(d_row, table.loc[i])
        table.at[i, "source"] = "manual" if diff else "default"
        if diff:
            changes.extend(diff)
            report.warnings.extend(pairing_limit_warnings(table.loc[i], limits))
    report.changes = pd.DataFrame(changes, columns=["station_id", "field", "default_value", "new_value"])
    return table, report


def load_pairing(
    stations: pd.DataFrame,
    default: pd.DataFrame,
    user_csv_path: str | None,
    *,
    use_user_pairing: bool,
    limits: PairingLimits = PairingLimits(),
    strict: bool = True,
):
    """Default pairing, or the validated user CSV when requested and present."""
    import os

    if use_user_pairing and user_csv_path and os.path.isfile(user_csv_path):
        table, report = apply_user_pairing(
            read_pairing_csv(user_csv_path), stations, default, limits=limits, strict=strict
        )
        return table, report
    return default.copy(), None


def method_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Stations per class and method, for a quick overview."""
    return (
        table.groupby(["class", "method"], observed=True).size().unstack(fill_value=0)
    )
