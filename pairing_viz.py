"""
Maps and Colab widgets for the station pairing.

``PairingSession`` holds the current pairing table. The deduction reads
``session.table``; the map and the editor are views on it and redraw when it
changes. Plotly/ipywidgets/cartopy are imported lazily, so ``pairing.py`` and
the tests do not need them.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import pairing as pr

CLASS_COLORS = {"RB": "#2ca02c", "UB": "#1f77b4", "TH": "#d62728", "unknown": "#8c564b"}
UNPAIRED_COLOR = "#9e9e9e"
RB_LINE = {"color": "#2ca02c", "width": 2.0}
UB_LINE = {"color": "#ff7f0e", "width": 1.2}
_NONE = "__none__"


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def filter_pairing(table: pd.DataFrame, countries=None, classes=None) -> pd.DataFrame:
    out = table
    if countries:
        out = out[out["country"].isin(set(countries))]
    if classes:
        out = out[out["class"].isin(set(classes))]
    return out


def is_unpaired(table: pd.DataFrame) -> pd.Series:
    """No usable reference: method none, or a non-RB station with neither RB nor UB pair."""
    no_ref = table["paired_rb_id"].isna() & table["paired_ub_id"].isna()
    return (table["method"] == "none") | ((table["class"] != "RB") & no_ref)


def _fmt(value, spec="", unit=""):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:{spec}}{unit}" if spec else f"{value}{unit}"


def station_hover_text(row) -> str:
    lines = [
        f"<b>{row['station_id']}</b> {row['station_name']}",
        f"class {row['class']} | area {row['area']} | type {row['type']}",
        f"altitude {_fmt(row['altitude'], '.0f', ' m')}",
        f"RB: {_fmt(row['paired_rb_id'])} {_fmt(row['paired_rb_name'])}"
        f" | {_fmt(row['rb_distance_km'], '.1f', ' km')} | Δh {_fmt(row['rb_altitude_diff_m'], '.0f', ' m')}",
    ]
    if row["class"] == "TH" or pd.notna(row["paired_ub_id"]):
        lines.append(
            f"UB: {_fmt(row['paired_ub_id'])} {_fmt(row['paired_ub_name'])}"
            f" | {_fmt(row['ub_distance_km'], '.1f', ' km')}"
        )
    lines.append(f"method {row['method']} | pairing {row['source']}")
    if str(row.get("comment", "") or "").strip():
        lines.append(f"comment: {row['comment']}")
    return "<br>".join(lines)


def _segments(table: pd.DataFrame, all_rows: pd.DataFrame, ref_col: str):
    """Lat/lon lists of station→reference lines, separated by None."""
    coords = all_rows.set_index("station_id")[["latitude", "longitude"]]
    lats, lons = [], []
    for sid, ref, lat, lon in zip(table["station_id"], table[ref_col], table["latitude"], table["longitude"]):
        if ref is None or (isinstance(ref, float) and np.isnan(ref)) or ref == sid or ref not in coords.index:
            continue
        lats += [lat, coords.at[ref, "latitude"], None]
        lons += [lon, coords.at[ref, "longitude"], None]
    return lats, lons


# ---------------------------------------------------------------------------
# Interactive map (Plotly)
# ---------------------------------------------------------------------------

def plot_station_pairing(
    table: pd.DataFrame,
    countries=None,
    classes=None,
    *,
    show_lines: bool = True,
    height: int = 650,
    zoom: float | None = None,
):
    """Plotly map of stations coloured by class with lines to their references.

    RB pairing lines are green and thicker, UB pairing lines (TH stations)
    orange and thinner. Stations without a usable reference are grey.
    """
    import plotly.graph_objects as go

    use_maplibre = hasattr(go, "Scattermap")
    Trace = go.Scattermap if use_maplibre else go.Scattermapbox
    view = filter_pairing(table, countries, classes)

    fig = go.Figure()
    if show_lines:
        for ref_col, style, name in (
            ("paired_rb_id", RB_LINE, "RB pairing"),
            ("paired_ub_id", UB_LINE, "UB pairing (TH)"),
        ):
            lats, lons = _segments(view, table, ref_col)
            fig.add_trace(Trace(lat=lats, lon=lons, mode="lines", line=style, name=name, hoverinfo="skip"))

    grey = is_unpaired(view)
    groups = [(cls, view[(view["class"] == cls) & ~grey], CLASS_COLORS[cls]) for cls in ("RB", "UB", "TH", "unknown")]
    groups.append(("No suitable pair", view[grey], UNPAIRED_COLOR))
    for name, rows, color in groups:
        if rows.empty:
            continue
        fig.add_trace(
            Trace(
                lat=rows["latitude"].tolist(),
                lon=rows["longitude"].tolist(),
                mode="markers",
                marker={"size": 9, "color": color},
                name=name,
                text=[station_hover_text(r) for _, r in rows.iterrows()],
                customdata=rows[["station_id"]].to_numpy(),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    if len(view):
        center = {"lat": float(view["latitude"].median()), "lon": float(view["longitude"].median())}
        span = max(np.ptp(view["latitude"]), np.ptp(view["longitude"]))
        zoom = zoom if zoom is not None else (3 if span > 15 else 5 if span > 4 else 7)
    else:
        center, zoom = {"lat": 50.0, "lon": 10.0}, 3
    map_key = "map" if use_maplibre else "mapbox"
    fig.update_layout(
        **{map_key: {"style": "carto-voyager" if use_maplibre else "carto-positron", "center": center, "zoom": zoom}},
        height=height,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        title=f"Station pairing: {len(view)} stations, {int(grey.sum())} without suitable pair",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01, "bgcolor": "rgba(255,255,255,0.7)"},
    )
    return fig


# ---------------------------------------------------------------------------
# Static map (Cartopy)
# ---------------------------------------------------------------------------

def plot_station_pairing_static(
    table: pd.DataFrame,
    countries=None,
    classes=None,
    *,
    ax=None,
    figsize=(10, 9),
    annotate: bool = False,
    savefile: str | None = None,
):
    """Cartopy version of the pairing map with a legend. Returns ``(fig, ax)``."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    view = filter_pairing(table, countries, classes)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()})
    else:
        fig = ax.figure
    for feature, alpha in ((cfeature.LAND, 0.3), (cfeature.OCEAN, 0.3), (cfeature.COASTLINE, 0.6), (cfeature.BORDERS, 0.6)):
        ax.add_feature(feature, alpha=alpha)

    for ref_col, style, ls in (("paired_rb_id", RB_LINE, "-"), ("paired_ub_id", UB_LINE, "--")):
        lats, lons = _segments(view, table, ref_col)
        lat_arr = np.array([np.nan if v is None else v for v in lats], dtype=float)
        lon_arr = np.array([np.nan if v is None else v for v in lons], dtype=float)
        ax.plot(lon_arr, lat_arr, ls, color=style["color"], linewidth=style["width"] * 0.6,
                alpha=0.8, transform=ccrs.PlateCarree())

    grey = is_unpaired(view)
    handles = [
        Line2D([], [], color=RB_LINE["color"], linestyle="-", label="RB pairing"),
        Line2D([], [], color=UB_LINE["color"], linestyle="--", label="UB pairing (TH)"),
    ]
    for cls in ("RB", "UB", "TH", "unknown"):
        rows = view[(view["class"] == cls) & ~grey]
        if rows.empty:
            continue
        ax.scatter(rows["longitude"], rows["latitude"], s=18, color=CLASS_COLORS[cls],
                   edgecolors="black", linewidths=0.3, transform=ccrs.PlateCarree(), zorder=3)
        handles.append(Line2D([], [], marker="o", linestyle="", color=CLASS_COLORS[cls], label=f"{cls} ({len(rows)})"))
    rows = view[grey]
    if not rows.empty:
        ax.scatter(rows["longitude"], rows["latitude"], s=24, color=UNPAIRED_COLOR, marker="X",
                   edgecolors="black", linewidths=0.3, transform=ccrs.PlateCarree(), zorder=4)
        handles.append(Line2D([], [], marker="X", linestyle="", color=UNPAIRED_COLOR, label=f"No suitable pair ({len(rows)})"))
    if annotate:
        for _, r in view.iterrows():
            ax.annotate(f"{r['station_id']}\n{r['method']} ({r['source']})", (r["longitude"], r["latitude"]),
                        fontsize=5, transform=ccrs.PlateCarree())
    if len(view):
        pad = 1.0
        ax.set_extent([view["longitude"].min() - pad, view["longitude"].max() + pad,
                       view["latitude"].min() - pad, view["latitude"].max() + pad], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.2)
    gl.top_labels = gl.right_labels = False
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.8)
    n_manual = int((view["source"] == "manual").sum())
    ax.set_title(f"Station pairing ({len(view)} stations, {n_manual} manual)")
    if savefile:
        fig.savefig(savefile, bbox_inches="tight", dpi=200)
    return fig, ax


# ---------------------------------------------------------------------------
# Session: the pairing table the deduction reads
# ---------------------------------------------------------------------------

class PairingSession:
    """Current pairing plus the default it can be reset to.

    Every change (editor or CSV) replaces ``table`` and notifies listeners,
    which redraw the map. With ``autosave_path`` each edit is also written to
    that CSV, so ``USE_USER_PAIRING = True`` picks it up in a later session.
    """

    def __init__(self, stations, default, table=None, *, limits=pr.PairingLimits(), autosave_path=None):
        self.stations = stations
        self.default = default
        self.table = (table if table is not None else default).copy()
        self.limits = limits
        self.autosave_path = autosave_path
        self._listeners = []

    def subscribe(self, callback):
        self._listeners.append(callback)

    def _changed(self, save=True):
        if save and self.autosave_path:
            pr.export_pairing_csv(self.table, self.autosave_path)
        for callback in list(self._listeners):
            callback(self.table)

    def edit(self, station_id, *, rb_id, ub_id, method=None, comment=None):
        self.table, warnings = pr.set_manual_pair(
            self.table, self.stations, station_id, rb_id=rb_id, ub_id=ub_id,
            method=method, comment=comment, limits=self.limits,
        )
        self._changed()
        return warnings

    def reset_station(self, station_id):
        self.table = pr.reset_station(self.table, self.default, station_id)
        self._changed()

    def reset_all(self):
        self.table = pr.reset_all(self.default)
        self._changed()

    def load_csv(self, path, strict=True):
        table, report = pr.apply_user_pairing(
            pr.read_pairing_csv(path), self.stations, self.default, limits=self.limits, strict=strict
        )
        self.table = table
        self._changed(save=False)
        return report

    def export_csv(self, path):
        return pr.export_pairing_csv(self.table, path)


def download_file(path: str) -> None:
    """Browser download in Colab; elsewhere just print the path."""
    if in_colab():
        from google.colab import files

        files.download(path)
    else:
        print(f"Saved to {os.path.abspath(path)}")


def upload_pairing_csv(dest_path: str) -> str | None:
    """Ask for a CSV upload in Colab and store it as ``dest_path``."""
    if not in_colab():
        print(f"Not in Colab. Place the edited file at {os.path.abspath(dest_path)} instead.")
        return dest_path if os.path.isfile(dest_path) else None
    from google.colab import files

    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded.")
        return None
    name, content = next(iter(uploaded.items()))
    with open(dest_path, "wb") as handle:
        handle.write(content)
    print(f"Stored {name} as {dest_path}")
    return dest_path


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

def pairing_map_widget(session: PairingSession, *, countries=None, on_select=None):
    """Map with country/class filters, redrawn when the session changes."""
    import ipywidgets as w
    import plotly.graph_objects as go

    all_countries = sorted(session.table["country"].dropna().unique())
    country_box = w.SelectMultiple(
        options=all_countries, value=tuple(c for c in (countries or all_countries) if c in all_countries),
        description="Country", rows=6, layout={"width": "220px"},
    )
    class_boxes = {cls: w.Checkbox(value=True, description=cls, indent=False, layout={"width": "70px"}) for cls in pr.CLASSES}

    if in_colab():
        try:
            from google.colab import output as colab_output

            colab_output.enable_custom_widget_manager()
        except Exception:
            pass

    def _figure():
        classes = [c for c, box in class_boxes.items() if box.value] + ["unknown"]
        return plot_station_pairing(session.table, list(country_box.value), classes)

    try:
        fig_widget = go.FigureWidget(_figure())
    except Exception:
        fig_widget = None
    out = w.Output()

    def _attach_clicks():
        if fig_widget is None or on_select is None:
            return
        for trace in fig_widget.data:
            if trace.mode == "markers":
                trace.on_click(lambda tr, pts, _s: pts.point_inds and on_select(tr.customdata[pts.point_inds[0]][0]))

    def redraw(*_):
        fig = _figure()
        if fig_widget is None:
            with out:
                out.clear_output(wait=True)
                fig.show()
            return
        fig_widget.data = []
        fig_widget.add_traces(list(fig.data))
        fig_widget.layout.title = fig.layout.title
        _attach_clicks()

    country_box.observe(redraw, names="value")
    for box in class_boxes.values():
        box.observe(redraw, names="value")
    session.subscribe(lambda _table: redraw())
    _attach_clicks()
    if fig_widget is None:
        redraw()
    filters = w.HBox([country_box, w.VBox([w.HTML("<b>Class</b>"), *class_boxes.values()])])
    return w.VBox([filters, fig_widget if fig_widget is not None else out])


def pairing_editor_widget(session: PairingSession, *, export_path: str | None = None):
    """Form to choose the paired RB/UB of one station, save, reset and export."""
    import ipywidgets as w

    def _label(row):
        return f"{row['station_id']} | {row['station_name']} [{row['class']}]"

    labels = {_label(r): r["station_id"] for _, r in session.table.iterrows()}
    station_box = w.Combobox(
        placeholder="Type a station id or name", options=sorted(labels), ensure_option=True,
        description="Station", layout={"width": "650px"},
    )
    rb_dd = w.Dropdown(description="Paired RB", layout={"width": "650px"})
    ub_dd = w.Dropdown(description="Paired UB", layout={"width": "650px"})
    method_dd = w.Dropdown(
        options=[("auto (from pairing)", ""), ("none: exclude station from the deduction", "none")],
        description="Method", layout={"width": "450px"},
    )
    comment_box = w.Text(description="Comment", layout={"width": "650px"})
    info = w.HTML()
    status = w.Output()
    save_btn = w.Button(description="Save", button_style="success", icon="save")
    reset_btn = w.Button(description="Reset station", icon="undo")
    reset_all_btn = w.Button(description="Reset all", button_style="warning", icon="refresh")
    export_btn = w.Button(description="Export CSV", icon="download")
    state = {"station": None}

    def _options(station_id, ref_class):
        cand = pr.candidate_references(session.stations, station_id, ref_class, limits=session.limits)
        none_label = "none / force Option 2" if ref_class == "RB" else "none"
        opts = [(none_label, _NONE)]
        for _, c in cand.iterrows():
            flag = "" if c["within_limits"] else "  (outside limits)"
            opts.append((
                f"{c['station_id']} | {c['station_name']} | {c['distance_km']:.1f} km | "
                f"Δh {_fmt(c['altitude_diff_m'], '.0f', ' m')}{flag}",
                c["station_id"],
            ))
        return opts

    def _show(station_id):
        state["station"] = station_id
        row = session.table.set_index("station_id").loc[station_id]
        if row["class"] == "RB":
            rb_dd.options = [(f"{station_id} (itself)", station_id), ("none / force Option 2", _NONE)]
        else:
            rb_dd.options = _options(station_id, "RB")
        valid_rb = {v for _, v in rb_dd.options}
        rb_dd.value = row["paired_rb_id"] if row["paired_rb_id"] in valid_rb else _NONE
        ub_dd.options = _options(station_id, "UB")
        valid_ub = {v for _, v in ub_dd.options}
        ub_dd.value = row["paired_ub_id"] if row["paired_ub_id"] in valid_ub else _NONE
        ub_dd.disabled = row["class"] != "TH"
        method_dd.value = "none" if row["method"] == "none" else ""
        comment_box.value = str(row["comment"] or "")
        info.value = station_hover_text(dict(row) | {"station_id": station_id})

    def select(station_id):
        for label, sid in labels.items():
            if sid == station_id:
                station_box.value = label
                return

    def _on_station(change):
        sid = labels.get(change["new"])
        if sid:
            _show(sid)

    station_box.observe(_on_station, names="value")

    def _run(action):
        with status:
            status.clear_output()
            try:
                action()
            except pr.PairingValidationError as exc:
                print(exc)

    def _save(_):
        sid = state["station"]
        if sid is None:
            with status:
                print("Choose a station first.")
            return

        def action():
            rb = None if rb_dd.value == _NONE else rb_dd.value
            ub = None if ub_dd.value == _NONE or ub_dd.disabled else ub_dd.value
            warnings = session.edit(sid, rb_id=rb, ub_id=ub, method=method_dd.value or None, comment=comment_box.value)
            for msg in warnings:
                print("Warning:", msg)
            row = session.table.set_index("station_id").loc[sid]
            print(f"Saved {sid}: method {row['method']} (manual).")
            if session.autosave_path:
                print(f"Pairing written to {session.autosave_path}")
            _show(sid)

        _run(action)

    def _reset(_):
        sid = state["station"]
        if sid is None:
            return

        def action():
            session.reset_station(sid)
            print(f"{sid} reset to the default pairing.")
            _show(sid)

        _run(action)

    def _reset_all(_):
        def action():
            session.reset_all()
            print("All stations reset to the default pairing.")
            if state["station"]:
                _show(state["station"])

        _run(action)

    def _export(_):
        def action():
            path = export_path or session.autosave_path or pr.USER_PAIRING_FILENAME
            session.export_csv(path)
            print(f"Exported {len(session.table)} stations to {path}")
            download_file(path)

        _run(action)

    save_btn.on_click(_save)
    reset_btn.on_click(_reset)
    reset_all_btn.on_click(_reset_all)
    export_btn.on_click(_export)
    box = w.VBox([
        station_box, info, rb_dd, ub_dd, method_dd, comment_box,
        w.HBox([save_btn, reset_btn, reset_all_btn, export_btn]), status,
    ])
    box.select_station = select
    return box


def pairing_app(session: PairingSession, *, countries=None, export_path=None):
    """Map and editor together; clicking a station opens it in the editor."""
    import ipywidgets as w

    editor = pairing_editor_widget(session, export_path=export_path)
    map_box = pairing_map_widget(session, countries=countries, on_select=editor.select_station)
    return w.VBox([
        w.HTML("<b>Click a station on the map or type it below, choose its references and press Save.</b>"),
        map_box, editor,
    ])
