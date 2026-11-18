"""Plots of the simplified Dust deduction notebook (section 8).

The plotting functions are the ones of ``Dust_deduction_tool_full copy.ipynb``;
the ``show_*`` functions take a :class:`dust_tool.DustTool` and only expose the
options a user may want to change.
"""
from __future__ import annotations

import math
import os
from typing import List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

import exceedance_report as er
import pairing_viz as pv

try:
    from IPython.display import display
except ImportError:  # plain Python: print instead
    display = print

COMPLIANCE_COLOURS = {
    "meets without deduction": "forestgreen",
    "meets only with deduction": "darkorange",
    "needs additional measures": "red",
}
LV_MAP_CLASSES = [
    ("meets without deduction", "forestgreen"),
    ("meets only with deduction", "darkorange"),
    ("exceeds even with deduction", "red"),
]
STATUS_LABELS = {
    "met": "meets without deduction",
    "met after deduction": "meets only with deduction",
    "still exceeded": "needs additional measures",
}
STATUS_CELL_STYLE = {
    "meets without deduction": "background-color: #c6efce",
    "meets only with deduction": "background-color: #ffd8a8",
    "needs additional measures": "background-color: #ffc7ce",
}


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------
def plot_interactive_station_map(
    df,
    color_column,
    size_column=None,
    color_type="auto",             # 'auto', 'continuous_scale', 'discrete_sequence'
    colorscale="Viridis",
    zoom=5,
    center_lat=None,
    center_lon=None,
    map_title=None,
    map_subtitle=None,
    colorbar_title=" ",
    marker_size_range=(8, 20),     # (min_px, max_px); min_px is the dot size of stations with value 0
    n_discrete_colors=10,
    size_legend_min=None,
    size_legend_max=None,
    size_legend_steps=4,
    color_min=None,
    color_max=None,
    color_label=None,              # readable name of color_column in the hover
    size_label=None,               # readable name of size_column in the hover and size legend
    hover_extra=None,              # extra hover rows: [(column, label, format), ...]
    marker_opacity=0.85,
    separate_duplicates=True,      # shift stations with identical coordinates slightly apart
):
    """Interactive Plotly map of stations.

    Uses MapLibre ``scatter_map`` with Carto Voyager tiles: 'open-street-map'
    tiles return 403 in Colab (missing Referer).
    """
    import plotly.express as px
    import plotly.graph_objects as go

    use_maplibre = hasattr(px, "scatter_map")
    scatter = px.scatter_map if use_maplibre else px.scatter_mapbox
    LegendTrace = go.Scattermap if use_maplibre else go.Scattermapbox
    trace_type = "scattermap" if use_maplibre else "scattermapbox"
    style_key = "map_style" if use_maplibre else "mapbox_style"
    style_value = "carto-voyager" if use_maplibre else "carto-positron"

    center_lat = 50.0 if center_lat is None else center_lat
    center_lon = 10.0 if center_lon is None else center_lon

    plotting_df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    color_col_to_plot = color_column
    color_label = color_label or color_column
    size_label = size_label or size_column

    if size_column is not None:
        # Large dots first, so the small ones are drawn on top and are not hidden.
        plotting_df = plotting_df.sort_values(size_column, ascending=False)
    if separate_duplicates:
        # Sampling points of the same station share coordinates; ~1 km apart they can be told apart when zooming in.
        n_dup = plotting_df.groupby(["Latitude", "Longitude"]).cumcount()
        plotting_df["Longitude"] = plotting_df["Longitude"] + 0.012 * n_dup

    plot_kwargs = {
        "lat": "Latitude", "lon": "Longitude", "zoom": zoom,
        "center": {"lat": center_lat, "lon": center_lon},
        "hover_name": "Samplingpoint", style_key: style_value, "opacity": marker_opacity,
    }
    is_color_numeric = pd.api.types.is_numeric_dtype(plotting_df[color_column])
    continuous = color_type == "continuous_scale" or (color_type == "auto" and is_color_numeric)

    if continuous:
        plot_kwargs["color"] = color_col_to_plot
        plot_kwargs["color_continuous_scale"] = colorscale
        if color_min is not None or color_max is not None:
            plot_kwargs["range_color"] = [
                color_min if color_min is not None else plotting_df[color_column].min(),
                color_max if color_max is not None else plotting_df[color_column].max(),
            ]
    elif color_type == "discrete_sequence" or (color_type == "auto" and not is_color_numeric):
        if is_color_numeric:
            color_col_to_plot = color_column + "_binned"
            plotting_df[color_col_to_plot] = pd.cut(
                plotting_df[color_column], bins=n_discrete_colors,
                labels=[f"{i}-{i + 1}" for i in range(n_discrete_colors)], include_lowest=True,
            )
        plot_kwargs["color"] = color_col_to_plot
        plot_kwargs["color_discrete_sequence"] = colorscale if isinstance(colorscale, list) else px.colors.qualitative.Plotly

    fig = scatter(plotting_df, **plot_kwargs)
    if colorbar_title and continuous:
        fig.update_coloraxes(colorbar=dict(title=colorbar_title, tickformat=".0f"))

    min_px, max_px = marker_size_range
    if size_column is not None:
        size_values = pd.to_numeric(plotting_df[size_column], errors="coerce").fillna(0).clip(lower=0)
        size_top = float(size_legend_max) if size_legend_max is not None else max(float(size_values.max()), 1.0)

        def to_pixels(values):
            # Explicit pixel sizes: every station, also with value 0, gets at least min_px.
            frac = np.sqrt(np.clip(np.asarray(values, dtype=float) / size_top, 0, 1))
            return min_px + (max_px - min_px) * frac

        fig.update_traces(marker=dict(size=to_pixels(size_values).tolist(), sizemode="diameter", sizeref=1),
                          selector=dict(type=trace_type))
    else:
        fig.update_traces(marker=dict(size=min_px), selector=dict(type=trace_type))

    hover_items = []
    if size_column is not None:
        hover_items.append((size_column, size_label, ".0f"))
    if color_column and is_color_numeric:
        hover_items.append((color_column, color_label, ".1f"))
    hover_items += [item for item in (hover_extra or []) if item[0] in plotting_df.columns]
    if hover_items:
        fig.update_traces(customdata=plotting_df[[col for col, _, _ in hover_items]].to_numpy(),
                          selector=dict(type=trace_type))
        lines = ["<b>%{hovertext}</b>"]
        lines += [f"{label}: %{{customdata[{i}]:{fmt}}}" for i, (_, label, fmt) in enumerate(hover_items)]
        lines.append("<extra></extra>")
        fig.update_traces(hovertemplate="<br>".join(lines), selector=dict(type=trace_type))

    fig.update_layout(title_text=map_title, title_x=0.5, title_y=0.97, margin=dict(t=110 if map_subtitle else 60))
    if map_subtitle:
        fig.add_annotation(text=map_subtitle, xref="paper", yref="paper", x=0.5, y=1.0, yshift=10,
                           showarrow=False, font=dict(size=12, color="gray"), xanchor="center", yanchor="bottom")

    if size_column is not None:
        smin = 0.0 if size_legend_min is None else float(size_legend_min)
        smax = size_top
        if smax < smin:
            smin, smax = smax, smin
        legend_values = [smin] if np.isclose(smin, smax) else list(np.linspace(smin, smax, max(2, size_legend_steps)))
        for v in legend_values:
            fig.add_trace(LegendTrace(
                lat=[center_lat], lon=[center_lon], mode="markers",
                marker=dict(size=float(to_pixels([v])[0]), color="blue"), name=f"{v:.0f}",
                legendgroup="size-legend", showlegend=True, visible="legendonly", hoverinfo="skip",
            ))
        fig.update_layout(legend=dict(
            title=f"{size_label}<br><i>(click a value to show its dot size)</i>", itemsizing="trace",
            yanchor="bottom", y=0.01, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)",
        ))
        fig.update_traces(legendrank=100, selector=dict(legendgroup="size-legend"))
    return fig


def plot_station_timeseries(
    station_name,
    obs_df,
    station_column="Samplingpoint",
    time_column="day",
    observed_pollutant_col="daily_mean",
    corrected_pollutant_col="corrected_pollutant",
    cams_dust_col="cams_dust",
    dust_flag_col="dust_flag",
    altitude_col="Altitude",
    exceedance_threshold=50.0,
    cams_dust_threshold=5.0,
    pollutant="PM10",
    year=2025,
    figsize=(12, 5),
):
    """Observed and dust-corrected PM (top) and CAMS dust (bottom) at one station; dates read as UTC."""
    required = [station_column, time_column, observed_pollutant_col, corrected_pollutant_col, cams_dust_col]
    missing = [col for col in required if col not in obs_df.columns]
    if missing:
        raise ValueError(f"Missing columns for the time series: {missing}")

    station_data = obs_df.loc[obs_df[station_column] == station_name].copy()
    if station_data.empty:
        raise ValueError(f"Station {station_name!r} is not in the observations.")
    station_data[time_column] = pd.to_datetime(station_data[time_column], errors="coerce", utc=True)
    year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    next_year = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    station_data = station_data.loc[
        (station_data[time_column] >= year_start) & (station_data[time_column] < next_year)
    ].sort_values(time_column)
    if station_data.empty:
        raise ValueError(f"Station {station_name!r} has no data for {year}.")

    altitude = None
    if altitude_col in station_data.columns:
        known = pd.to_numeric(station_data[altitude_col], errors="coerce").dropna()
        if not known.empty:
            altitude = float(known.iloc[0])

    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=(3, 1), sharex=True)
    axes[0].plot(station_data[time_column], station_data[observed_pollutant_col],
                 label=f"Observed {pollutant}", linewidth=1.2)
    axes[0].plot(station_data[time_column], station_data[corrected_pollutant_col],
                 label=f"Corrected {pollutant}", linewidth=1.2, linestyle="--")
    if dust_flag_col in station_data.columns:
        dust_days = station_data.loc[station_data[dust_flag_col].fillna(False).astype(bool)]
        if not dust_days.empty:
            axes[0].scatter(dust_days[time_column], dust_days[observed_pollutant_col],
                            color="crimson", s=20, label="Flagged dust day", zorder=3)
    axes[0].axhline(exceedance_threshold, color="royalblue", linestyle=":",
                    label=f"Daily threshold ({exceedance_threshold:g} µg/m³)")
    alt_label = f" | altitude {altitude:.0f} m" if altitude is not None else ""
    axes[0].set_title(f"{station_name} | {pollutant} | {year}{alt_label}")
    axes[0].set_ylabel(f"{pollutant} (µg/m³)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(station_data[time_column], station_data[cams_dust_col],
                 color="forestgreen", linewidth=1.3, label="CAMS dust")
    axes[1].axhline(cams_dust_threshold, color="forestgreen", linestyle=":",
                    label=f"Dust flag threshold ({cams_dust_threshold:g} µg/m³)")
    axes[1].set_ylabel("CAMS dust (µg/m³)")
    axes[1].set_xlabel("Date (UTC)")
    axes[1].set_xlim(year_start, next_year)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig, axes


def _map_figure_widget_class():
    import plotly.graph_objects as go

    class MapFigureWidget(go.FigureWidget):
        """Ignores the read-only ``map._derived`` viewport updates Plotly sends on zoom or pan."""

        def _perform_plotly_relayout(self, relayout_data):
            if isinstance(relayout_data, dict):
                relayout_data = {key: value for key, value in relayout_data.items()
                                 if not (key.split(".", 1)[0].startswith("map") and key.endswith("._derived"))}
            return super()._perform_plotly_relayout(relayout_data)

    return MapFigureWidget


def map_timeseries_clickable_plot(
    obs_df,
    year,
    exceedance_threshold,
    cams_dust_threshold,
    pollutant,
    dataset="",
    station_col="Samplingpoint",
    time_col="day",
    observed_pollutant_col="daily_mean",
    corrected_pollutant_col="corrected_pollutant",
    cams_dust_col="cams_dust",
    dust_flag_col="dust_flag",
    altitude_col="Altitude",
    lat_col="Latitude",
    lon_col="Longitude",
    value_col="daily_mean",
    cmap_name="plasma",
    radius_range=(6, 16),
    vmin=None,
    vmax=None,
    legend_title=None,
    legend_bins=5,
    legend_round=1,
    zoom=None,
):
    """Clickable station map (annual mean of ``value_col``); a click or the dropdown shows the station time series."""
    import plotly.graph_objects as go
    from ipywidgets import HTML, Dropdown, Output, VBox

    needed = [station_col, time_col, lat_col, lon_col, value_col,
              observed_pollutant_col, corrected_pollutant_col, cams_dust_col]
    missing = [column for column in needed if column not in obs_df.columns]
    if missing:
        raise ValueError(f"Missing columns for the interactive map: {missing}")

    df = obs_df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    df = df.loc[(df[time_col] >= start) & (df[time_col] < end)].copy()
    for column in (lat_col, lon_col, value_col):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[station_col, lat_col, lon_col, value_col])
    if df.empty:
        return VBox([HTML(f"No stations with coordinates and values for {year}.")])

    summary = (
        df.groupby(station_col, observed=True, sort=False)
        .agg(average=(value_col, "mean"), latitude=(lat_col, "first"), longitude=(lon_col, "first"))
        .reset_index()
    )
    low = float(summary["average"].min()) if vmin is None else float(vmin)
    high = float(summary["average"].max()) if vmax is None else float(vmax)
    if high <= low:
        high = low + 1
    rmin, rmax = radius_range
    if rmin <= 0 or rmax < rmin:
        raise ValueError("radius_range must have increasing positive values")
    lat_span = summary["latitude"].max() - summary["latitude"].min()
    lon_span = summary["longitude"].max() - summary["longitude"].min()
    if zoom is None:
        zoom = 4 if lat_span > 12 or lon_span > 15 else 6
    center = {"lat": float(summary["latitude"].median()), "lon": float(summary["longitude"].median())}

    # Explicit pixel sizes keep low-concentration stations visible in the widget renderer.
    size_fraction = np.clip((summary["average"].to_numpy(dtype=float) - low) / (high - low), 0, 1)
    station_trace = go.Scattermap(
        lat=summary["latitude"].to_numpy(dtype=float).tolist(),
        lon=summary["longitude"].to_numpy(dtype=float).tolist(),
        mode="markers",
        text=summary[station_col].astype(str).tolist(),
        customdata=summary[[station_col]].to_numpy(),
        hovertemplate="<b>%{text}</b><br>Average " + pollutant + ": %{marker.color:.2f} µg/m³<extra></extra>",
        marker={
            "symbol": "circle", "size": (rmin + (rmax - rmin) * size_fraction).tolist(),
            "sizemode": "diameter", "sizeref": 1,
            "color": summary["average"].to_numpy(dtype=float).tolist(), "colorscale": cmap_name,
            "cmin": low, "cmax": high, "opacity": 0.9, "showscale": True,
            "colorbar": {"title": legend_title or f"Average {pollutant} (µg/m³)",
                         "tickvals": np.linspace(low, high, max(2, legend_bins)),
                         "tickformat": f".{max(0, legend_round)}f"},
        },
    )
    fig = go.Figure(station_trace)
    fig.update_layout(map={"style": "carto-voyager", "center": center, "zoom": zoom}, height=560,
                      title=f"Observed {dataset} stations annual average {pollutant} ({year})",
                      margin={"l": 0, "r": 0, "t": 45, "b": 0})

    try:
        from google.colab import output as colab_output
        colab_output.enable_custom_widget_manager()
    except ImportError:
        pass
    try:
        map_widget = _map_figure_widget_class()(fig)
    except ImportError as exc:
        raise ImportError("For the clickable Plotly map run %pip install 'plotly>=6' anywidget ipywidgets") from exc

    plot_out = Output()
    selector = Dropdown(options=[("Select a station", None)] + [(str(n), n) for n in summary[station_col]],
                        value=None, description="Station:", layout={"width": "550px"})

    def show_timeseries(station_name):
        with plot_out:
            plot_out.clear_output(wait=True)
            if station_name is None:
                return
            try:
                station_fig, _ = plot_station_timeseries(
                    station_name=station_name, obs_df=df, station_column=station_col, time_column=time_col,
                    observed_pollutant_col=observed_pollutant_col, corrected_pollutant_col=corrected_pollutant_col,
                    cams_dust_col=cams_dust_col, dust_flag_col=dust_flag_col, altitude_col=altitude_col,
                    exceedance_threshold=exceedance_threshold, cams_dust_threshold=cams_dust_threshold,
                    pollutant=pollutant, year=year,
                )
                display(station_fig)
                plt.close(station_fig)
            except Exception as exc:
                print(f"Cannot display station {station_name!r}: {exc}")

    selector.observe(lambda change: show_timeseries(change["new"]), names="value")

    def on_map_click(trace, points, state):
        if points.point_inds:
            station_name = trace.customdata[points.point_inds[0]][0]
            if selector.value == station_name:
                show_timeseries(station_name)
            else:
                selector.value = station_name

    for trace in map_widget.data:
        if trace.type == "scattermap":
            trace.on_click(on_map_click)

    return VBox([
        HTML(f"<b>{len(summary)} stations loaded.</b> Click a station on the map to show its daily "
             "time series, or choose it from the list."),
        map_widget, selector, plot_out,
    ])


def plot_exceedance_maps_discrete(
    df: pd.DataFrame,
    columns_to_plot: Sequence[str],
    titles: Sequence[str],
    vmax: Union[float, Sequence[float]],
    cbar_text: Union[str, Sequence[str]],
    cmap_name: Union[str, Sequence[str]] = "gist_heat_r",
    n_colors: Union[int, Sequence[int]] = 10,
    extent: Optional[List[float]] = None,
    gridline: bool = False,
    cbar_mode: Literal["single", "each"] = "single",
    cbar_orientation: Literal["vertical", "horizontal"] = "vertical",
    cbar_tick_mode: Literal["bounds", "centers"] = "bounds",
    cbar_size: str = "2.0%",
    cbar_pad: str = "1.5%",
    cbar_shrink: float = 0.85,
    cbar_pad_single: float = 0.05,
    marker_size: float = 15,
    marker_alpha: float = 0.8,
    edgecolor: str = "black",
    edge_lw: float = 0.5,
    coast_alpha: float = 0.6,
    borders_alpha: float = 0.6,
    land_alpha: float = 0.3,
    ocean_alpha: float = 0.3,
    lakes_alpha: float = 0.3,
    title_fontsize: int = 12,
    label_fontsize: int = 9,
    figsize_per_panel: Tuple[float, float] = (5.0, 6.0),
    n_cols: Optional[int] = None,
    categories: Optional[Sequence[Optional[Sequence[Tuple[str, str]]]]] = None,
    hspace: float = 0.55,
    show: bool = True,
    savefile: Optional[str] = None,
    dpi: int = 200,
):
    """Static station maps with discrete colour levels, one panel per column.

    ``vmax``, ``cbar_text``, ``cmap_name`` and ``n_colors`` take one value or one per panel.
    ``categories`` has one entry per panel: None for a numeric map, or (value, colour) pairs
    for a categorical map, which gets a legend instead of a colour bar. ``n_cols`` sets maps
    per row (an incomplete last row is centred). NaN values are drawn as hollow circles.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    assert len(columns_to_plot) == len(titles), "Number of columns must match number of titles."
    n_panels = len(columns_to_plot)

    def per_panel(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            assert len(value) == n_panels, "Per-panel options need one value per column."
            return list(value)
        return [value] * n_panels

    vmaxs, cmap_names, ncols, cbar_texts = map(per_panel, (vmax, cmap_name, n_colors, cbar_text))
    cats_list = list(categories) if categories is not None else [None] * n_panels
    assert len(cats_list) == n_panels, "categories needs one entry per column."

    cmaps = [plt.get_cmap(name, n) for name, n in zip(cmap_names, ncols)]
    bounds_list = [np.linspace(0, v, n + 1) for v, n in zip(vmaxs, ncols)]
    norms = [mcolors.BoundaryNorm(b, c.N) for b, c in zip(bounds_list, cmaps)]

    if n_cols is None or n_cols >= n_panels:
        fig, axs = plt.subplots(1, n_panels, figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
                                subplot_kw={"projection": ccrs.PlateCarree()}, dpi=dpi)
        if n_panels == 1:
            axs = [axs]
    else:
        n_rows = int(np.ceil(n_panels / n_cols))
        fig = plt.figure(figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows), dpi=dpi)
        # Two grid columns per map, so a last row with fewer maps can be centred.
        gs = fig.add_gridspec(n_rows, 2 * n_cols, hspace=hspace, wspace=0.4)
        axs = []
        for k in range(n_panels):
            row, col_idx = divmod(k, n_cols)
            in_row = min(n_cols, n_panels - row * n_cols)
            start = (n_cols - in_row) + 2 * col_idx
            axs.append(fig.add_subplot(gs[row, start:start + 2], projection=ccrs.PlateCarree()))

    scatters = []
    for i, (col, title) in enumerate(zip(columns_to_plot, titles)):
        ax = axs[i]
        cats = cats_list[i]
        if cats is None:
            sc = ax.scatter(df["Longitude"], df["Latitude"], c=df[col], s=marker_size, cmap=cmaps[i], norm=norms[i],
                            transform=ccrs.PlateCarree(), alpha=marker_alpha, edgecolors=edgecolor, linewidth=edge_lw)
        else:
            colours = df[col].map(dict(cats)).fillna("lightgrey")
            sc = ax.scatter(df["Longitude"], df["Latitude"], c=colours.tolist(), s=marker_size,
                            transform=ccrs.PlateCarree(), alpha=marker_alpha, edgecolors=edgecolor, linewidth=edge_lw)
            counts = df[col].value_counts()
            handles = [Line2D([], [], marker="o", ls="", markerfacecolor=colour, markeredgecolor=edgecolor,
                              markersize=8, label=f"{value} ({counts.get(value, 0)})") for value, colour in cats]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                      fontsize=label_fontsize + 1, frameon=False)
        scatters.append(sc)

        ax.add_feature(cfeature.COASTLINE, alpha=coast_alpha)
        ax.add_feature(cfeature.BORDERS, alpha=borders_alpha)
        ax.add_feature(cfeature.LAND, alpha=land_alpha)
        ax.add_feature(cfeature.OCEAN, alpha=ocean_alpha)
        ax.add_feature(cfeature.LAKES, alpha=lakes_alpha)
        if extent is None:
            buffer_val = 1.5
            ax.set_extent([df["Longitude"].min() - buffer_val, df["Longitude"].max() + buffer_val,
                           df["Latitude"].min() - buffer_val, df["Latitude"].max() + buffer_val],
                          crs=ccrs.PlateCarree())
        else:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        ax.set_title(title, fontsize=title_fontsize)
        labels = i == 0 or cbar_mode == "each"
        gl = ax.gridlines(draw_labels=labels, dms=False, x_inline=False, y_inline=False)
        if labels:
            gl.top_labels = False
            gl.right_labels = False
            gl.bottom_labels = True
            gl.left_labels = True
            gl.xlabel_style = {"size": label_fontsize}
            gl.ylabel_style = {"size": label_fontsize}
        try:
            gl.xlines = gridline
            gl.ylines = gridline
        except Exception:
            pass

    if cbar_tick_mode == "bounds":
        ticks_list = bounds_list
    elif cbar_tick_mode == "centers":
        ticks_list = [(b[:-1] + b[1:]) / 2.0 for b in bounds_list]
    else:
        raise ValueError("cbar_tick_mode must be 'bounds' or 'centers'.")

    if cbar_mode == "single":
        sm = plt.cm.ScalarMappable(norm=norms[0], cmap=cmaps[0])
        sm.set_array([])
        pad = cbar_pad_single if cbar_orientation == "vertical" else 0.08
        cbar = fig.colorbar(sm, ax=axs, orientation=cbar_orientation, shrink=cbar_shrink, pad=pad,
                            ticks=ticks_list[0], extend="max", spacing="proportional")
        cbar.set_label(cbar_texts[0], fontsize=label_fontsize + 2)
        cbar.ax.tick_params(labelsize=label_fontsize)
    elif cbar_mode == "each":
        size_ratio = float(cbar_size.replace("%", "")) / 100.0
        pad_ratio = float(cbar_pad.replace("%", "")) / 100.0
        # Colour bar axes are placed by hand to avoid interaction with the GeoAxes.
        for ax, sc, ticks, text, cats in zip(axs, scatters, ticks_list, cbar_texts, cats_list):
            if cats is not None:
                continue
            ax.apply_aspect()  # map box after cartopy fixed the aspect ratio
            pos = ax.get_position()
            if cbar_orientation == "vertical":
                cax = fig.add_axes([pos.x1 + pos.width * pad_ratio, pos.y0, pos.width * size_ratio, pos.height])
            elif cbar_orientation == "horizontal":
                height = pos.height * size_ratio
                cax = fig.add_axes([pos.x0, pos.y0 - pos.height * pad_ratio - height, pos.width, height])
            else:
                raise ValueError("cbar_orientation must be 'vertical' or 'horizontal'.")
            cb = plt.colorbar(sc, cax=cax, orientation=cbar_orientation, ticks=ticks,
                              extend="max", spacing="proportional")
            cb.set_label(text, fontsize=label_fontsize + 1)
            cb.ax.tick_params(labelsize=label_fontsize)
    else:
        raise ValueError("cbar_mode must be 'single' or 'each'.")

    if savefile:
        fig.savefig(savefile, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tables behind the maps
# ---------------------------------------------------------------------------
def _compliance(measured_ok: pd.Series, adjusted_ok: pd.Series, worst="needs additional measures") -> np.ndarray:
    return np.select([measured_ok, adjusted_ok], ["meets without deduction", "meets only with deduction"], default=worst)


def exceedance_map_table(station_table, df_target_year, countries, daily_limit, allowed_exceedances=18,
                         hide_zero_days_removed=True) -> pd.DataFrame:
    """Station table of the selected countries with the columns of the exceedance maps (section 8.3)."""
    table = station_table[station_table["Samplingpoint"].str.startswith(tuple(c + "/" for c in countries))].copy()
    removed = table["Exceedance_days"] - table["NonDust_exceedance"]
    table["Exceedance_days_removed"] = removed.where(removed > 0) if hide_zero_days_removed else removed

    adjusted_col = "adjusted_value" if "adjusted_value" in df_target_year.columns else "corrected_pollutant"
    rows = df_target_year[df_target_year["daily_mean"] > daily_limit]
    means = rows.groupby(rows["Samplingpoint"].astype(str)).agg(measured=("daily_mean", "mean"),
                                                                adjusted=(adjusted_col, "mean"))
    key = table["Samplingpoint"].astype(str)
    table["Exceedance_mean_measured"] = key.map(means["measured"])
    table["Exceedance_mean_removed"] = key.map(means["measured"] - means["adjusted"])
    table["Daily_LV_compliance"] = _compliance(table["Exceedance_days"] <= allowed_exceedances,
                                               table["NonDust_exceedance"] <= allowed_exceedances,
                                               worst="exceeds even with deduction")
    return table


def annual_map_table(station_table, annual_limit) -> pd.DataFrame:
    table = station_table.assign(
        Average_pollutant_removed=station_table["Average_pollutant"] - station_table["Average_pollutant_dust_removed"])
    table["Annual_LV_compliance"] = _compliance(table["Average_pollutant"] <= annual_limit,
                                                table["Average_pollutant_dust_removed"] <= annual_limit,
                                                worst="exceeds even with deduction")
    return table


def station_compliance_tables(tool, allowed_exceedances=18, countries=None):
    """Per-station compliance (``er.station_compliance``) and the display table of section 8.4."""
    countries = countries or tool.settings.countries
    daily = tool.df_target_year[
        tool.df_target_year["Samplingpoint"].astype(str).str.startswith(tuple(c + "/" for c in countries))]
    if tool.station_pairing is not None:
        daily = er.add_station_info(daily, tool.station_pairing)
    stations = er.station_compliance(daily, daily_limit=tool.daily_limit, annual_limit=tool.annual_limit,
                                     allowed_exceedances=allowed_exceedances)

    exc = daily[pd.to_numeric(daily["daily_mean"], errors="coerce") > tool.daily_limit]
    exc_means = exc.groupby(exc["Samplingpoint"].astype(str)).agg(measured=("daily_mean", "mean"),
                                                                  adjusted=("corrected_pollutant", "mean"))
    names = (tool.station_pairing.drop_duplicates("station_id").set_index("station_id")["station_name"]
             if tool.station_pairing is not None else pd.Series(dtype=str))
    s = tool.settings
    annual_col = f"Annual LV ({tool.annual_limit} µg/m³)"
    daily_col = f"Daily LV (≤ {allowed_exceedances} days > {tool.daily_limit} µg/m³)"
    sp = stations["Samplingpoint"].astype(str)
    station_id = sp.map(er.base_station_id)
    report = pd.DataFrame({
        "Country": station_id.str[:2],
        "Station id": sp,
        "Station name": station_id.map(names),
        "Class": stations["station_class"],
        "Annual mean measured (µg/m³)": stations["annual_mean_measured"],
        "Annual mean adjusted (µg/m³)": stations["annual_mean_adjusted"],
        "Mean on exceedance days measured (µg/m³)": sp.map(exc_means["measured"]),
        "Mean on exceedance days adjusted (µg/m³)": sp.map(exc_means["adjusted"]),
        "Exceedance days measured": stations["exceedances_measured"],
        "Exceedance days adjusted": stations["exceedances_adjusted"],
        annual_col: stations["annual_lv_status"].map(STATUS_LABELS),
        daily_col: stations["daily_lv_status"].map(STATUS_LABELS),
    }).sort_values(["Country", "Station id"], ignore_index=True)
    print(f"{s.pollutant} {s.year}: daily LV {tool.daily_limit} µg/m³ (max {allowed_exceedances} days), "
          f"annual LV {tool.annual_limit} µg/m³. {len(report)} stations.")
    return stations, report, (annual_col, daily_col)


# ---------------------------------------------------------------------------
# Notebook steps (section 8)
# ---------------------------------------------------------------------------
def show_corrected_annual_map(tool, size_legend_max=19, colour_max_factor=2.5, zoom=3):
    """Interactive map: colour = annual mean after the deduction, size = exceedance days left (section 8.1)."""
    s = tool.settings
    fig = plot_interactive_station_map(
        df=tool.station_table, color_column="Average_pollutant_dust_removed", size_column="NonDust_exceedance",
        color_type="continuous_scale", colorscale="viridis", zoom=zoom,
        map_title=f"{s.pollutant} {s.year} after subtracting the natural (dust) contribution",
        map_subtitle=(f"Colour = annual mean after the deduction (annual limit value {tool.annual_limit} µg/m³)<br>"
                      f"Dot size = days above the daily limit value ({tool.daily_limit} µg/m³) that remain after the deduction"),
        colorbar_title="Annual mean<br>after deduction<br>(µg/m³)",
        color_label="Annual mean after deduction (µg/m³)", size_label="Exceedance days after deduction",
        hover_extra=[("Average_pollutant", "Annual mean measured (µg/m³)", ".1f"),
                     ("Exceedance_days", "Exceedance days measured", ".0f"),
                     ("Dust_exceedance_days", "Exceedance days removed by the deduction", ".0f")],
        size_legend_min=0, size_legend_max=size_legend_max, size_legend_steps=4, marker_size_range=(8, 30),
        color_min=0, color_max=colour_max_factor * tool.annual_limit,
    )
    print(f"{len(tool.station_table)} stations on the map")
    fig.show(renderer="colab" if tool.in_colab else None)


def show_clickable_timeseries_map(tool, vmin=0, vmax=25, legend_bins=7, radius_range=(8, 16)):
    """Clickable map with the time series of the selected station (section 8.2)."""
    s = tool.settings
    widget = map_timeseries_clickable_plot(
        obs_df=tool.df_target_year, year=s.year, pollutant=s.pollutant, dataset=tool.dataset,
        exceedance_threshold=tool.daily_limit, cams_dust_threshold=s.cams_dust_threshold,
        vmin=vmin, vmax=vmax, legend_title=f"{s.pollutant} (µg/m³)", legend_bins=legend_bins,
        legend_round=0, radius_range=radius_range,
    )
    display(widget)


def show_station_timeseries(tool, station=0):
    """Static time series; ``station`` is a Samplingpoint id or its position in the station table."""
    if not isinstance(station, str):
        station = tool.station_table["Samplingpoint"].unique()[station]
    s = tool.settings
    return plot_station_timeseries(station, tool.df_target_year, year=s.year, exceedance_threshold=tool.daily_limit,
                                   cams_dust_threshold=s.cams_dust_threshold, pollutant=s.pollutant)


def show_exceedance_maps(tool, allowed_exceedances=18, colormap="inferno_r", diff_colormap="YlGnBu",
                         hide_zero_days_removed=True):
    """Exceedance days and mean on exceedance days, observed and removed, plus daily LV compliance (section 8.3)."""
    s = tool.settings
    table = exceedance_map_table(tool.station_table, tool.df_target_year, s.countries, tool.daily_limit,
                                 allowed_exceedances, hide_zero_days_removed)
    removed = table["Exceedance_days"] - table["NonDust_exceedance"]
    # Scale of "days removed" from the stations with at least one day removed; most have 0.
    q95 = max(1, int(np.ceil(removed[removed > 0].quantile(0.95)))) if (removed > 0).any() else 1
    n_days_removed = min(10, q95)
    vmax_days_removed = int(np.ceil(q95 / n_days_removed)) * n_days_removed
    vmax_conc_removed = max(1.0, float(np.ceil(table["Exceedance_mean_removed"].quantile(0.95))))
    removed_title = "Exceedance days removed\n(observed − after deduction" + ("; hollow = 0)" if hide_zero_days_removed else ")")
    plot_exceedance_maps_discrete(
        table,
        ["Exceedance_days", "Exceedance_days_removed", "Exceedance_mean_measured", "Exceedance_mean_removed",
         "Daily_LV_compliance"],
        [f"Exceedance days {s.year} (observed)", removed_title,
         f"Mean {s.pollutant} on exceedance days (observed)",
         f"{s.pollutant} removed on exceedance days\n(observed − after deduction)",
         f"Daily limit value compliance\n(≤ {allowed_exceedances} days above {tool.daily_limit} µg/m³)"],
        vmax=[2 * allowed_exceedances, vmax_days_removed, 2 * tool.daily_limit, vmax_conc_removed, 1],
        n_colors=[6, n_days_removed, 10, 10, 1],
        cmap_name=[colormap, diff_colormap, colormap, diff_colormap, colormap],
        cbar_text=["days", "days removed", "µg/$m^3$", "µg/$m^3$ removed", ""],
        categories=[None, None, None, None, LV_MAP_CLASSES],
        n_cols=2, cbar_mode="each", cbar_orientation="horizontal", cbar_size="3.5%", cbar_pad="12%",
        figsize_per_panel=(5.5, 5.0),
    )


def show_annual_maps(tool, colormap="inferno_r", diff_colormap="YlGnBu", n_colors=10):
    """Observed annual mean, annual mean removed by the deduction and annual LV compliance (section 8.3)."""
    s = tool.settings
    table = annual_map_table(tool.station_table, tool.annual_limit)
    diff_vmax = max(1.0, float(np.ceil(table["Average_pollutant_removed"].quantile(0.95))))
    plot_exceedance_maps_discrete(
        table,
        ["Average_pollutant", "Average_pollutant_removed", "Annual_LV_compliance"],
        [f"{s.year}: Average {s.pollutant} observed", f"{s.year}: {s.pollutant} removed by dust deduction",
         f"Annual limit value compliance\n(annual mean ≤ {tool.annual_limit} µg/m³)"],
        vmax=[tool.daily_limit, diff_vmax, 1], n_colors=[n_colors, n_colors, 1],
        cmap_name=[colormap, diff_colormap, colormap],
        cbar_text=["µg/$m^3$", "µg/$m^3$ removed (observed − after deduction)", ""],
        categories=[None, None, LV_MAP_CLASSES], cbar_mode="each", cbar_orientation="horizontal",
        gridline=False, cbar_size="3.5%", cbar_pad="12%", marker_size=10,
    )


def show_compliance_report(tool, allowed_exceedances=18, countries=None, save_csv=True):
    """Coloured station compliance table (section 8.4); returns the per-station compliance for the maps."""
    stations, report, (annual_col, daily_col) = station_compliance_tables(tool, allowed_exceedances, countries)
    styled = report.set_index(["Country", "Station id"]).style.format(precision=1, na_rep="–")
    colour_cells = styled.map if hasattr(styled, "map") else styled.applymap  # pandas < 2.1 only has applymap
    display(colour_cells(lambda v: STATUS_CELL_STYLE.get(v, ""), subset=[annual_col, daily_col]))
    if save_csv:
        s = tool.settings
        path = os.path.join(tool.output_dir,
                            f"{'_'.join(countries or s.countries)}_{s.pollutant}_{s.year}_station_compliance.csv")
        report.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig: Excel shows µ and ≤
        pv.download_file(path)
    return stations


def show_compliance_maps(tool, stations, allowed_exceedances=18):
    """Annual and daily limit value status of the compliance table on a map (section 8.4)."""
    table = stations.copy()
    if not {"Latitude", "Longitude"} <= set(table.columns):
        table = table.merge(tool.station_table[["Samplingpoint", "Latitude", "Longitude"]], on="Samplingpoint", how="left")
    table = table.dropna(subset=["Latitude", "Longitude"])
    table["Annual_LV"] = table["annual_lv_status"].map(STATUS_LABELS)
    table["Daily_LV"] = table["daily_lv_status"].map(STATUS_LABELS)
    classes = list(COMPLIANCE_COLOURS.items())
    plot_exceedance_maps_discrete(
        table, ["Annual_LV", "Daily_LV"],
        [f"Annual limit value ({tool.annual_limit} µg/m³)",
         f"Daily limit value (≤ {allowed_exceedances} days > {tool.daily_limit} µg/m³)"],
        vmax=1, n_colors=1, cbar_text="", categories=[classes, classes],
        cbar_mode="each", marker_size=25, figsize_per_panel=(6.0, 6.0),
    )


def show_threshold_sweep(tool, consistency, suggestion, group="country", min_share=0.9, min_days=10,
                         plot_cols=5, save=False):
    """Heatmap, per-group curves and suggested threshold of the threshold sweep (section 7.2.1)."""
    s = tool.settings
    thr_col = "dust_threshold_ug_m3"
    groups = sorted(consistency[group].unique())
    thresholds = sorted(consistency[thr_col].unique())
    share = consistency.pivot(index=group, columns=thr_col, values="share_confirmed").reindex(index=groups, columns=thresholds)
    days = consistency.pivot(index=group, columns=thr_col, values="cams_dust_days").reindex(index=groups, columns=thresholds)
    sugg = suggestion.set_index(group).reindex(groups)
    suggested = sugg["model_threshold_ug_m3"]
    figs = {}

    fig, ax = plt.subplots(figsize=(0.9 * len(thresholds) + 3, 0.4 * len(groups) + 1.5))
    im = ax.imshow(share.values, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    for i, g in enumerate(groups):
        for j in range(len(thresholds)):
            sh, n = share.iat[i, j], days.iat[i, j]
            if pd.isna(sh):
                continue
            ax.text(j, i, f"{sh:.0%}", ha="center", va="center", fontsize=8)
            if n < min_days:
                ax.add_patch(mpatches.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///", edgecolor="grey", lw=0.5))
        if pd.notna(suggested[g]):
            j = thresholds.index(suggested[g])
            ax.add_patch(mpatches.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="black", lw=2))
    ax.set_xticks(range(len(thresholds)), [f"{t:g}" for t in thresholds])
    ax.set_yticks(range(len(groups)), groups)
    ax.set_xlabel("CAMS dust threshold (µg/m³)")
    ax.set_title(f"Share of CAMS dust days confirmed by the measurements ({s.pollutant} {s.year})\n"
                 f"black frame = suggested threshold, hatched = fewer than {min_days} dust days")
    fig.colorbar(im, ax=ax, label="share confirmed", format=PercentFormatter(1.0))
    figs["heatmap"] = fig
    plt.show()

    ncols = min(plot_cols, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharex=True, sharey=True, squeeze=False)
    x = np.array(thresholds)
    for ax, g in zip(axes.flat, groups):
        ax2 = ax.twinx()
        ax2.bar(x, days.loc[g].fillna(0), width=0.6, color="lightsteelblue", alpha=0.7)
        ax2.tick_params(labelsize=7, colors="steelblue")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        enough = (days.loc[g] >= min_days).to_numpy()
        ax.plot(x, share.loc[g], "-", color="C0")
        ax.plot(x[enough], share.loc[g].to_numpy()[enough], "o", color="C0")
        ax.plot(x[~enough], share.loc[g].to_numpy()[~enough], "o", color="C0", mfc="white")
        ax.axhline(min_share, ls="--", color="red", lw=1)
        sg = suggested[g]
        if pd.notna(sg):
            ax.axvline(sg, color="green", lw=1.5)
        ax.set_title(f"{g}: {f'{sg:g} µg/m³' if pd.notna(sg) else 'not reached'}", fontsize=10)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.4, 1.03)
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    fig.supxlabel("CAMS dust threshold (µg/m³)")
    fig.supylabel("share confirmed (line)")
    fig.suptitle(f"Share confirmed and number of CAMS dust days (bars, right axis) per {group}\n"
                 f"red = {min_share:.0%} criterion, green = suggested threshold, "
                 f"open marker = fewer than {min_days} dust days")
    fig.tight_layout()
    figs["per_group"] = fig
    plt.show()

    few = sugg["thresholds_tested"] <= 2
    heights = suggested.fillna(max(thresholds))
    colors = np.where(suggested.isna(), "lightgrey", np.where(few, "orange", "seagreen"))
    fig, ax = plt.subplots(figsize=(0.6 * len(groups) + 5, 4.5))
    bars = ax.bar(groups, heights, color=colors, edgecolor="black", lw=0.5)
    for bar, sg in zip(bars, suggested):
        if pd.isna(sg):
            bar.set_hatch("//")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{sg:g}" if pd.notna(sg) else "–", ha="center", fontsize=8)
    ax.axhline(s.cams_dust_threshold, ls="--", color="black", lw=1)
    ax.set_ylabel("suggested threshold (µg/m³)")
    ax.set_title(f"Lowest CAMS dust threshold with ≥ {min_share:.0%} confirmed dust days ({s.pollutant} {s.year})")
    ax.legend(handles=[
        mpatches.Patch(color="seagreen", label="suggested threshold"),
        mpatches.Patch(color="orange", label="suggested, few dust days (≤ 2 thresholds usable)"),
        mpatches.Patch(facecolor="lightgrey", hatch="//", label=f"not reached up to {max(thresholds):g} µg/m³"),
        plt.Line2D([], [], ls="--", color="black", label=f"current cams_dust_threshold ({s.cams_dust_threshold:g})"),
    ], fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    figs["suggested"] = fig
    plt.show()

    if save:
        for name, f in figs.items():
            path = os.path.join(tool.output_dir, f"threshold_sweep_{name}_{s.pollutant}_{tool.dataset}_{s.year}.png")
            f.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Saved {path}")
