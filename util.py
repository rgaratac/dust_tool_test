import numpy as np
import pandas as pd
import xarray as xr 
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import ipywidgets  
from IPython.display import display
from ipywidgets import Output, HTML, VBox
from ipyleaflet import Map, CircleMarker, LayersControl, basemaps, LayersControl, LegendControl
from ipywidgets import VBox, HTML, Output
from typing import List, Optional, Literal, Sequence, Tuple
import matplotlib.pyplot as plt  # used by plot_station_timeseries if it shows figures
import matplotlib.cm as cm
import plotly.express as px 
import plotly.graph_objects as go

def plot_interactive_station_map(
    df,
    color_column,
    size_column=None,
    color_type='auto',             # 'auto', 'continuous_scale', 'discrete_sequence'
    colorscale='Viridis',
    zoom=5,
    center_lat=None,
    center_lon=None,
    map_title=None,
    map_subtitle=None,
    colorbar_title=' ',            # Default to single space to show no title if unwanted
    marker_size_range=[8, 20],     # [min_px, max_px] — note: px.scatter_mapbox only uses size_max directly
    n_discrete_colors=10,
    # NEW: user-controlled size legend values (your smin/smax)
    size_legend_min=None,
    size_legend_max=None, 
    size_legend_steps=4
):
    """
    Plots an interactive scatter mapbox of stations using Plotly Express.

    Args:
        df (pd.DataFrame): DataFrame containing station data.
        color_column (str): Column for coloring.
        size_column (str, optional): Column for sizing bubbles.
        color_type (str): 'auto' | 'continuous_scale' | 'discrete_sequence'.
        colorscale (str or list/dict): Colors for color dim.
        zoom (int): Map zoom.
        center_lat (float, optional), center_lon (float, optional): Map center.
        map_title (str, optional), map_subtitle (str, optional).
        colorbar_title (str): Title for colorbar (continuous color).
        marker_size_range (list[int,int]): [min_px, max_px] desired display sizes. Only max is used by px directly.
        n_discrete_colors (int): #bins if discrete sequence on numeric col.
        size_legend_min (float, optional): Minimum data value to display in size legend.
        size_legend_max (float, optional): Maximum data value to display in size legend.
        size_legend_steps (int): Number of size bubbles to show in legend.
    """

    # Center dict may be None; we'll also provide a safe fallback for legend-only traces later
    center_dict = {'lat': center_lat, 'lon': center_lon} if (center_lat is not None and center_lon is not None) else None

    plotting_df = df.copy()
    color_col_to_plot = color_column

    plot_kwargs = {
        'lat': 'Latitude',
        'lon': 'Longitude',
        'size': size_column,
        'zoom': zoom,
        'center': center_dict,
        'hover_name': 'Samplingpoint',
        'mapbox_style': 'open-street-map',
        'size_max': marker_size_range[1],  # px uses only size_max to set the top bubble size
    }

    # --- Color handling ---
    is_color_numeric = pd.api.types.is_numeric_dtype(plotting_df[color_column])

    if color_type == 'continuous_scale' or (color_type == 'auto' and is_color_numeric):
        plot_kwargs['color'] = color_col_to_plot
        plot_kwargs['color_continuous_scale'] = colorscale

    elif color_type == 'discrete_sequence' or (color_type == 'auto' and not is_color_numeric):
        # If user forced discrete sequence on a numeric column, bin it:
        if is_color_numeric:
            color_col_to_plot = color_column + '_binned'
            plotting_df[color_col_to_plot] = pd.cut(
                plotting_df[color_column],
                bins=n_discrete_colors,
                labels=[f'{i}-{i+1}' for i in range(n_discrete_colors)],
                include_lowest=True
            )
        plot_kwargs['color'] = color_col_to_plot
        plot_kwargs['color_discrete_sequence'] = colorscale if isinstance(colorscale, list) else px.colors.qualitative.Plotly

    # --- Create figure ---
    fig = px.scatter_mapbox(plotting_df, **plot_kwargs)

    # --- Continuous color: set colorbar title + tick format ---
    if colorbar_title and (color_type == 'continuous_scale' or (color_type == 'auto' and is_color_numeric)):
        fig.update_coloraxes(colorbar=dict(title=colorbar_title, tickformat=".2f"))

    # --- Hover formatting for size and numeric color ---
    custom_cols, custom_idx = [], {}
    pos = 0
    if size_column is not None:
        custom_cols.append(size_column); custom_idx['size'] = pos; pos += 1
    if color_column and is_color_numeric:
        custom_cols.append(color_column); custom_idx['color'] = pos; pos += 1

    if custom_cols:
        fig.update_traces(
            customdata=plotting_df[custom_cols].to_numpy(),
            selector=dict(type="scattermapbox")
        )
        hover_lines = ["<b>%{hovertext}</b>"]
        if 'size' in custom_idx:
            hover_lines.append(f"{size_column}: %{{customdata[{custom_idx['size']}]:.0f}}")
        if 'color' in custom_idx:
            hover_lines.append(f"{color_column}: %{{customdata[{custom_idx['color']}]:.2f}}")
        hover_lines.append("<extra></extra>")
        fig.update_traces(
            hovertemplate="<br>".join(hover_lines),
            selector=dict(type="scattermapbox")
        )

    # --- Title & subtitle ---
    fig.update_layout(
        title_text=map_title,
        title_x=0.5,
        title_y=0.95,
    )
    if map_subtitle:
        fig.add_annotation(
            text=map_subtitle,
            xref="paper", yref="paper",
            x=0.5, y=0.91,
            showarrow=False,
            font=dict(size=12, color="gray"),
            xanchor="center", yanchor="top"
        )

    # ===============================
    # Size legend (legend-only traces)
    # ===============================
    if size_column is not None:
        # 1) Determine representative data values to show in legend
        data_min = float(np.nanmin(plotting_df[size_column]))
        data_max = float(np.nanmax(plotting_df[size_column]))

        smin = data_min if size_legend_min is None else float(size_legend_min)
        smax = data_max if size_legend_max is None else float(size_legend_max)
        if smax < smin:
            smin, smax = smax, smin  # swap if user accidentally inverted

        # Choose values to display
        size_values = [smin] if np.isclose(smin, smax) else list(np.linspace(smin, smax, max(2, size_legend_steps)))

        # 2) Reuse PX-computed sizing parameters so legend matches map
        main_trace = next((t for t in fig.data if t.type == "scattermapbox"), None)
        if main_trace and hasattr(main_trace, "marker") and hasattr(main_trace.marker, "sizeref"):
            computed_sizeref = main_trace.marker.sizeref
            sizemode = getattr(main_trace.marker, "sizemode", "diameter")
            sizemin = getattr(main_trace.marker, "sizemin", None)
        else:
            # Fallback if not found (rare)
            computed_sizeref = None
            sizemode = "diameter"
            sizemin = None

        # 3) Safe coordinates for legend-only traces
        if center_dict is not None:
            lat0, lon0 = center_dict["lat"], center_dict["lon"]
        else:
            # Fallback to first row coords (required even if legendonly)
            lat0 = float(plotting_df['Latitude'].iloc[0])
            lon0 = float(plotting_df['Longitude'].iloc[0])

        # 4) Add one trace per size value (legend-only)
        for v in size_values:
            fig.add_trace(go.Scattermapbox(
                lat=[lat0], lon=[lon0],
                mode="markers",
                marker=dict(
                    size=v,                        # IMPORTANT: pass the DATA value, not pixels
                    sizemode=sizemode,
                    sizeref=computed_sizeref,     # reuse mapping from main trace
                    sizemin=sizemin,
                    color="blue"
                ),
                name=f"{v:.2f}",
                legendgroup="size-legend",
                showlegend=True,
                visible="legendonly",
                hoverinfo="skip"
            ))

        # 5) Make legend respect per-trace marker sizes
        fig.update_layout(
            legend=dict(
                title=f"{size_column}",
                itemsizing="trace",            # <-- KEY so the legend icon size matches the trace
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.6)"
            )
        )

        # Optional: ensure the size legend entries appear after color categories
        fig.update_traces(legendrank=100, selector=dict(legendgroup="size-legend"))
      

    return fig

def plot_station_timeseries(
    station_name: str,
    obs_df: pd.DataFrame,
    station_column: str = 'Samplingpoint',
    time_column: str = 'Start',
    observed_pollutant_col: str = 'observed_PM10',
    corrected_pollutant_col: str = 'corrected_PM10',
    cams_dust_col: str = 'cams_dust', 
    dust_flag_col: str = 'dust_flag',
    altitude_col : str='Altitude',
    exceedance_threshold: float = 50.0,
    cams_dust_threshold: float = 5.0,
    pollutant: str = 'PM10',
    # Plot controls
    year: int = 2024,
    figsize=(12, 5),
    title_fontsize: int = 12,
    label_fontsize: int = 10,
    legend_fontsize: int = 9,
):
    """
    Plot Pollutant (observed vs. corrected) and CAMS dust for a station.

    Parameters:
    -----------
    station_name : str
        The unique ID of the station to plot.
    obs_df : pd.DataFrame
        The DataFrame containing time series observations.
    station_column : str
        Column name in `obs_df` identifying the station.
    time_column : str
        Column name for timestamps/dates.
    observed_pollutant_col : str
        Column name for original measured concentrations.
    corrected_pollutant_col : str
        Column name for concentrations after dust subtraction.
    cams_dust_col : str
        Column name for the CAMS dust component values.
    dust_flag_col : str
        Boolean column indicating if a day is flagged as a dust event.
    altitude_col : str
        Column name for station altitude metadata.
    exceedance_threshold : float
        Threshold line for original pollutant (e.g., 50 µg/m³).
    cams_dust_threshold : float
        Threshold line for CAMS dust flagging.
    pollutant : str
        Name of the pollutant (e.g., 'PM10' or 'PM2.5') for labels.
    year : int
        The calendar year to display.
    """
    # --- Select data for this station ---
    station_mask = (obs_df[station_column] == station_name)
    station_data = obs_df.loc[station_mask].copy()
    if station_data.empty:
        raise ValueError(f"No data found for station '{station_name}' in obs_df.")

    # --- Normalize time column ---
    station_data[time_column] = pd.to_datetime(station_data[time_column], errors='coerce')
    t_min = pd.Timestamp(f'{year}-01-01')
    t_max = pd.Timestamp(f'{year}-12-31')
    station_data = station_data[(station_data[time_column] >= t_min) & (station_data[time_column] <= t_max)]

    if station_data.empty:
        raise ValueError(f"No rows for '{station_name}' within year {year}.")

    altitude = None
    if not obs_df.empty and station_column in obs_df.columns:
        alt_row = obs_df.loc[obs_df[station_column] == station_name, altitude_col]
        if not alt_row.empty and pd.notna(alt_row.iloc[0]):
            altitude = float(alt_row.iloc[0])

    fig, axs = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)

    # --- Upper panel ---
    axs[0].plot(
        station_data[time_column], station_data[observed_pollutant_col],
        marker='o', markersize=4, label=f'Original {pollutant}',
        linewidth=1.5, alpha=0.8
    )
    axs[0].plot(
        station_data[time_column], station_data[corrected_pollutant_col],
        marker='x', markersize=4, linestyle='--', label=f'Corrected {pollutant}',
        linewidth=1.5, alpha=0.8
    )

    if dust_flag_col in station_data.columns:
        dust_days = station_data[station_data[dust_flag_col].astype(bool)]
        if not dust_days.empty:
            axs[0].scatter(
                dust_days[time_column], dust_days[observed_pollutant_col],
                color='red', s=30, alpha=0.7, label='Dust days', zorder=5
            )

    axs[0].axhline(y=exceedance_threshold, color='blue', linestyle=':', alpha=0.6, label=f'Exceedance ({exceedance_threshold} µg/m³)')
    axs[0].set_ylabel(f'{pollutant} (µg/m³)', fontsize=label_fontsize)
    axs[0].grid(True, alpha=0.3)
    axs[0].tick_params(axis='x', labelsize=label_fontsize)
    axs[0].legend(loc='upper right', fontsize=legend_fontsize, framealpha=0.4)

    title_alt = f" | Altitude: {altitude:.0f} m" if altitude is not None else ""
    axs[0].set_title(f'{station_name} — {pollutant} Timeseries {year}{title_alt}', fontsize=title_fontsize, fontweight='bold')

    # --- Lower panel ---
    axs[1].plot(
        station_data[time_column], station_data[cams_dust_col],
        color='green', linewidth=2, alpha=0.7, label='CAMS Dust'
    )
    axs[1].axhline(y=cams_dust_threshold, color='green', linestyle=':', alpha=0.6, label=f'Dust threshold ({cams_dust_threshold} µg/m³)')
    axs[1].set_xlabel('Date', fontsize=label_fontsize)
    axs[1].set_ylabel('CAMS Dust (µg/m³)', fontsize=label_fontsize, color='green')
    axs[1].tick_params(axis='y', labelcolor='green', labelsize=label_fontsize)
    axs[1].tick_params(axis='x', labelsize=label_fontsize+1)
    axs[1].legend(loc='upper right', fontsize=legend_fontsize, framealpha=0.4)
    axs[1].set_xlim(t_min, t_max)

    plt.tight_layout()
    return fig, axs
def map_timeseries_clickable_plot(
    obs_df,
    year,
    exceedance_threshold,
    cams_dust_threshold, 
    pollutant,
    station_col    : str= 'Samplingpoint',
    time_col       : str= 'Start',
    observed_pollutant_col : str= 'daily_mean',
    corrected_pollutant_col: str= 'corrected_pm10',
    cams_dust_col     : str = 'cams_dust',
    dust_flag_col     : str = 'dust_flag',
    altitude_col      : str = 'Altitude', 
    lat_col           : str = 'Latitude', 
    lon_col           : str = 'Longitude',
    value_col         : str ='daily_mean',
    # --- Color & size controls ---
    cmap_name='plasma',           # any Matplotlib colormap name
    radius_range=(6, 16),         # (min_px, max_px)
    vmin=None,                    # NEW: Minimum value for color scale
    vmax=None,                    # NEW: Maximum value for color scale
    # --- Legend controls ---
    legend_title=None,            # defaults to f"Average {value_col}"
    legend_bins=5,                # number of bins (discrete steps to represent the continuous colormap)
    legend_round=1,               # decimals in legend labels
    legend_position='bottomright' # any Leaflet corner: 'topleft', 'topright', 'bottomleft', 'bottomright'
):
    """
    Build an ipyleaflet map with station markers colored & sized by the per-station
    average of `value_col`. Clicking a marker renders that station's time series.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Observation dataframe containing at least the columns specified by
        station_col, time_col, lat_col, lon_col, value_col.
    year : int
        Year passed to plot_station_timeseries (no filtering here).
    exceedance_threshold : float
        Passed to plot_station_timeseries.
    cams_dust_threshold : float
        Passed to plot_station_timeseries.
    station_col, time_col, lat_col, lon_col, value_col : str
        Column names to use for station ID, timestamp, latitude, longitude, and measured value.
    cmap_name : str
        Matplotlib colormap name for coloring markers.
    radius_range : (int, int)
        Pixel radius range for marker sizes (min, max).
    vmin, vmax : float, optional
        Manual color scale range. If None, derived from data.
    legend_title : str or None
        Title for the legend. If None, uses f"Average {value_col}".
    legend_bins : int
        Number of discrete color bins used to represent the continuous colormap.
    legend_round : int
        Number of decimals to format legend bin labels.
    legend_position : str
        Position of the legend on the map.
    """

    df = obs_df.copy()
    # Parse time column if present
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    # Coordinates per station (first non-null)
    coords = (
        df[[station_col, lat_col, lon_col]]
        .dropna(subset=[lat_col, lon_col])
        .drop_duplicates(subset=[station_col])
        .set_index(station_col)
    )

    # Per-station average of the value_col (for marker color & size)
    avg_col_name = f"avg_{value_col}"
    avg_vals = (
        df.groupby(station_col)[value_col]
          .mean()
          .rename(avg_col_name)
          .to_frame()
    )

    summary = (
        avg_vals.join(coords, how='inner')
                .dropna(subset=[lat_col, lon_col])
                .reset_index()
    )

    if summary.empty:
        return VBox([HTML("<b>Error: No stations with coordinates and values.</b>")])

    # -----------------------------
    # Helpers for color/size scaling
    # -----------------------------
    def color_for_value(val, vmin, vmax, cmap_name='plasma'):
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cmap_name)
        return mcolors.to_hex(cmap(norm(val)))

    def radius_for_value(val, vmin, vmax, rmin=6, rmax=16):
        return float(np.interp(val, [vmin, vmax], [rmin, rmax]))

    # Scale bounds (across stations' averages) - Apply manual override if provided
    actual_vmin = vmin if vmin is not None else float(summary[avg_col_name].min())
    actual_vmax = vmax if vmax is not None else float(summary[avg_col_name].max())

    # -----------------------------
    # Build the map (centered over median coords)
    # -----------------------------
    center_lat = float(summary[lat_col].median())
    center_lon = float(summary[lon_col].median())

    m = Map(center=(center_lat, center_lon), zoom=6, basemap={'url': 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'})
    m.add_control(LayersControl())

    # Output area for the time series below the map
    plot_out = Output()
    desc = HTML("<b>Click a station marker</b> to load its time series below.")

    # -----------------------------
    # Color legend (discrete bins from the continuous colormap)
    # -----------------------------
    legend_title = legend_title or f"Average {value_col}"
    # Bin edges & labels
    edges = np.linspace(actual_vmin, actual_vmax, legend_bins + 1)
    # Use midpoints to sample the colormap
    mids = (edges[:-1] + edges[1:]) / 2.0

    legend_items = {}
    for i, mid in enumerate(mids):
        label = f"{edges[i]:.{legend_round}f}–{edges[i+1]:.{legend_round}f}"
        legend_items[label] = color_for_value(mid, actual_vmin, actual_vmax, cmap_name=cmap_name)

    legend = LegendControl(legend_items, title=legend_title, position=legend_position)
    m.add(legend)

    # -----------------------------
    # Add markers (click → plot_station_timeseries)
    # -----------------------------
    rmin, rmax = radius_range
    for _, row in summary.iterrows():
        sp  = row[station_col]
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        val = float(row[avg_col_name])

        color  = color_for_value(val, actual_vmin, actual_vmax, cmap_name=cmap_name)
        radius = int(radius_for_value(val, actual_vmin, actual_vmax, rmin=rmin, rmax=rmax))

        marker = CircleMarker(
            location=(lat, lon),
            radius=radius,
            color=color,
            fill_color=color,
            fill_opacity=0.75,
            stroke=False
        )

        # Use a factory to bind the current station to the click handler
        def create_on_click_callback(current_sp):
            def on_click_callback(**kwargs):
                with plot_out:
                    plot_out.clear_output(wait=True)
                    try:
                        # Try to pass value_col if the function supports it
                        try:
                            fig_ts, axs = plot_station_timeseries(
                                station_name=current_sp,
                                obs_df=df,
                                year=year,
                                time_column =time_col,
                                cams_dust_col       = cams_dust_col,
                                dust_flag_col       = dust_flag_col,
                                altitude_col        = altitude_col,
                                exceedance_threshold=exceedance_threshold,
                                cams_dust_threshold =cams_dust_threshold,
                                observed_pollutant_col   = observed_pollutant_col,
                                corrected_pollutant_col  = corrected_pollutant_col,
                                figsize=(12, 5),
                                pollutant = pollutant
                            )
                        except TypeError:
                            # Fallback if plot_station_timeseries doesn't accept value_col
                            fig_ts, axs = plot_station_timeseries(
                                station_name=current_sp,
                                obs_df              = df,
                                year                = year,
                                time_column         = time_col,
                                cams_dust_col       = cams_dust_col,
                                dust_flag_col       = dust_flag_col,
                                altitude_col        = altitude_col,
                                exceedance_threshold= exceedance_threshold,
                                cams_dust_threshold = cams_dust_threshold,
                                observed_pollutant_col   = observed_pollutant_col,
                                corrected_pollutant_col  = corrected_pollutant_col,
                                figsize=(12, 5),
                                pollutant = pollutant
                            )
                        plt.show()
                    except Exception as e:
                        print(f"Cannot render station '{current_sp}': {e}")
            return on_click_callback

        marker.on_click(create_on_click_callback(sp))
        m.add_layer(marker)

    # -----------------------------
    # Display
    # -----------------------------
    return VBox([desc, m, plot_out])


def until_check():
    return "util.py has been imported"


def plot_station_pairing(pairing_table, countries=None, classes=None, static=False, **kwargs):
    """Station pairing map: Plotly (default) or Cartopy (``static=True``). See pairing_viz.py."""
    import pairing_viz

    if static:
        return pairing_viz.plot_station_pairing_static(pairing_table, countries, classes, **kwargs)
    return pairing_viz.plot_station_pairing(pairing_table, countries, classes, **kwargs)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# Removed: from mpl_toolkits.axes_grid1 import make_axes_locatable (not needed for the fix)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import List, Optional, Literal, Sequence, Tuple

def plot_exceedance_maps_discrete(
    df: pd.DataFrame,
    columns_to_plot: Sequence[str],
    titles: Sequence[str],
    vmax: float,
    cbar_text: str,
    cmap_name: str = "gist_heat_r",
    n_colors: int = 10,
    extent: Optional[List[float]] = None,
    gridline : bool = False ,
    # --- NEW OPTIONS ---
    cbar_mode: Literal["single", "each"] = "single",
    cbar_orientation: Literal["vertical", "horizontal"] = "vertical",
    cbar_tick_mode: Literal["bounds", "centers"] = "bounds",
    cbar_size: str = "2.0%",     # only used when cbar_mode='each'
    cbar_pad: str = "1.5%",      # only used when cbar_mode='each'
    cbar_shrink: float = 0.85,   # used for cbar_mode='single'
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
    show: bool = True,
    savefile: Optional[str] = None,
    dpi: int = 200,
):
    """
    Plot maps of exceedance data with discrete color levels and flexible colorbar options.

    Parameters
    ----------
    df : DataFrame
        Must contain 'Latitude', 'Longitude', and columns in `columns_to_plot`.
    columns_to_plot : list[str]
        Column names in df to plot.
    titles : list[str]
        Titles for each subplot.
    vmax : float
        Upper bound for the discrete color scale (lower bound is 0).
    cbar_text : str
        Colorbar label text.
    cmap_name : str
        Matplotlib colormap name, e.g., 'gist_heat_r'.
    n_colors : int
        Number of discrete levels (creates n_colors+1 boundaries from 0..vmax).
    extent : [min_lon, max_lon, min_lat, max_lat] or None
        Map extent. If None, computed dynamically from df with a small buffer.

    Notes
    -----
    - `cbar_mode="single"`: one shared colorbar spanning all axes (fig.colorbar).
    - `cbar_mode="each"`: a separate narrow colorbar per subplot,
      attached via axes_grid1's `make_axes_locatable`.
    - `cbar_orientation` applies to both modes ('vertical' or 'horizontal').
    """
    assert len(columns_to_plot) == len(titles), "Number of columns must match number of titles."

    # --- Discrete colormap & normalization ---
    cmap = plt.get_cmap(cmap_name, n_colors)
    bounds = np.linspace(0, vmax, n_colors + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # --- Figure & axes ---
    n_panels = len(columns_to_plot)
    fig_w = figsize_per_panel[0] * n_panels
    fig_h = figsize_per_panel[1]
    fig, axs = plt.subplots(
        1, n_panels,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=dpi
    )

    # Ensure axs is iterable for 1 panel
    if n_panels == 1:
        axs = [axs]

    scatters = []

    # --- Plot each panel ---
    for i, (col, title) in enumerate(zip(columns_to_plot, titles)):
        ax = axs[i]

        # Scatter
        sc = ax.scatter(
            df["Longitude"], df["Latitude"],
            c=df[col],
            s=marker_size, cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(),
            alpha=marker_alpha, edgecolors=edgecolor, linewidth=edge_lw
        )
        scatters.append(sc)

        # Map features
        ax.add_feature(cfeature.COASTLINE, alpha=coast_alpha)
        ax.add_feature(cfeature.BORDERS, alpha=borders_alpha)
        ax.add_feature(cfeature.LAND, alpha=land_alpha)
        ax.add_feature(cfeature.OCEAN, alpha=ocean_alpha)
        ax.add_feature(cfeature.LAKES, alpha=lakes_alpha)

        # Extent
        if extent is None:
            buffer_val = 1.5
            dynamic_extent = [
                df["Longitude"].min() - buffer_val, df["Longitude"].max() + buffer_val,
                df["Latitude"].min() - buffer_val, df["Latitude"].max() + buffer_val
            ]
            ax.set_extent(dynamic_extent, crs=ccrs.PlateCarree())
        else:
            ax.set_extent(extent, crs=ccrs.PlateCarree())


        ax.set_title(title, fontsize=title_fontsize)
        if i == 0 or cbar_mode == "each":
            gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
            gl.top_labels = False
            gl.right_labels = False
            gl.bottom_labels = True
            gl.left_labels = True
            # Hide grid lines (labels only)
            try:
                gl.xlines = gridline
                gl.ylines = gridline
            except Exception:
                pass
            gl.xlabel_style = {"size": label_fontsize}
            gl.ylabel_style = {"size": label_fontsize}
        else:
            gl = ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
            try:
                gl.xlines = gridline
                gl.ylines = gridline
            except Exception:
                pass
    # --- Colorbar ticks: bounds vs centers ---
    if cbar_tick_mode == "bounds":
        ticks = bounds
    elif cbar_tick_mode == "centers":
        ticks = (bounds[:-1] + bounds[1:]) / 2.0
    else:
        raise ValueError("cbar_tick_mode must be 'bounds' or 'centers'.")

    # --- Build colorbar(s) ---
    if cbar_mode == "single":
        # Use a ScalarMappable so the colorbar is independent of a specific artist
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        # Choose pad depending on orientation (slightly larger for horizontal)
        pad = cbar_pad_single if cbar_orientation == "vertical" else 0.08

        cbar = fig.colorbar(
            sm, ax=axs, orientation=cbar_orientation,
            shrink=cbar_shrink, pad=pad, ticks=ticks, extend="max", spacing="proportional"
        )
        cbar.set_label(cbar_text, fontsize=label_fontsize + 2)
        cbar.ax.tick_params(labelsize=label_fontsize)


    elif cbar_mode == "each":
        # Manually create and position colorbar axes for 'each' mode to avoid GeoAxes interaction
        for ax, sc in zip(axs, scatters):
            ax_pos = ax.get_position()  # Get the bounding box of the main axis
            cax = None

            # Convert string percentages to floats representing a fraction of the *axis* dimension
            cbar_size_ratio = float(cbar_size.replace('%', '')) / 100.0
            cbar_pad_ratio = float(cbar_pad.replace('%', '')) / 100.0

            if cbar_orientation == "vertical":
                # Calculate position for a vertical colorbar to the right of the main axis
                # x0, y0, width, height for fig.add_axes
                cax_width = ax_pos.width * cbar_size_ratio
                cax_x0 = ax_pos.x1 + (ax_pos.width * cbar_pad_ratio)
                cax_y0 = ax_pos.y0
                cax_height = ax_pos.height

                cax = fig.add_axes([cax_x0, cax_y0, cax_width, cax_height])

            elif cbar_orientation == "horizontal":
                # Calculate position for a horizontal colorbar below the main axis
                cax_height = ax_pos.height * cbar_size_ratio
                cax_y0 = ax_pos.y0 - (ax_pos.height * cbar_pad_ratio) - cax_height
                cax_x0 = ax_pos.x0
                cax_width = ax_pos.width

                cax = fig.add_axes([cax_x0, cax_y0, cax_width, cax_height])

            else:
                raise ValueError("cbar_orientation must be 'vertical' or 'horizontal'.")

            if cax is not None:
                cb = plt.colorbar(
                    sc, cax=cax, orientation=cbar_orientation,
                    ticks=ticks, extend="max", spacing="proportional"
                )
                cb.set_label(cbar_text, fontsize=label_fontsize + 1)
                cb.ax.tick_params(labelsize=label_fontsize)
    else:
        raise ValueError("cbar_mode must be 'single' or 'each'.")

    # Remove tight_layout for 'each' mode as it conflicts with manually placed axes.
    # It is usually still desired for 'single' mode with shared colorbar.
    # if cbar_mode == "single":
    #     # fig.tight_layout()

    if savefile:
        fig.savefig(savefile, bbox_inches="tight", dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)


########################################################
############ hourly data processing function ###########
########################################################

def eea_hourly_to_utc(series: pd.Series, source_tz: str = "CET") -> pd.Series:

    """
    Convert a pandas Series of datetimes to timezone-aware UTC, assuming the source is a fixed UTC+1.
    - Naive timestamps are localized to the fixed-offset zone 'Etc/GMT-1' (which represents UTC+1).
    - Already tz-aware timestamps are kept and then converted to UTC.
    - Non-parsable values become NaT and are preserved.
    """
    # Parse to datetime; invalids -> NaT
    s = pd.to_datetime(series, errors="coerce")

    def _localize_if_naive(x):
        if pd.isna(x):
            return x
        # naive -> localize to fixed UTC+1 (Etc/GMT-1), avoids DST issues
        return x.tz_localize("Etc/GMT-1") if x.tzinfo is None else x

    s = s.apply(_localize_if_naive)

    # Convert everything to UTC, resulting dtype: datetime64[ns, UTC]
    return s.dt.tz_convert("UTC")

def filter_daily_by_coverage(
    df_daily: pd.DataFrame,
    day_col: str = 'day',
    station_col: str = 'Samplingpoint',
    reference_year: int | None = None,
    min_pct: float = 75.0,
    keep_coverage_columns: bool = True
) -> pd.DataFrame:
    """
    Keep ALL rows for stations that have coverage >= min_pct IN the reference year.
    Example: if a station has >=75% coverage in 2024, keep its 2023 data as well since average window need longer rolling time.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily dataframe (one row per day per station).
    day_col : str
        Column with daily timestamp (date or datetime).
    station_col : str
        Station/sampling point identifier column.
    reference_year : int | None
        The year used to evaluate coverage. If None, coverage is evaluated
        across ALL years (station-level over the whole dataset).
    min_pct : float
        Coverage threshold (default 75.0).
    keep_coverage_columns : bool
        If True, merge per-(station, year) coverage metrics onto the output.

    Returns
    -------
    pd.DataFrame
        All rows for stations that meet the coverage criterion in the reference year.
        If keep_coverage_columns=True, includes coverage columns:
        ['year','unique_days','total_days','coverage_percentage','sufficient_coverage'].
    """
    df = df_daily.copy()

    # Normalize day and extract year
    df[day_col] = pd.to_datetime(df[day_col], utc=True, errors='coerce').dt.floor('D')
    df = df.dropna(subset=[day_col, station_col])
    df['year'] = df[day_col].dt.year

    # --- Compute coverage per (station, year) ---
    coverage = (
        df.groupby([station_col, 'year'])[day_col]
          .nunique()
          .rename('unique_days')
          .reset_index()
    )
    coverage['total_days'] = coverage['year'].apply(lambda y: 366 if calendar.isleap(y) else 365)
    coverage['coverage_percentage'] = (coverage['unique_days'] / coverage['total_days']) * 100.0
    coverage['sufficient_coverage'] = coverage['coverage_percentage'] >= min_pct

    # --- Decide which stations qualify ---
    if reference_year is not None:
        # Stations that meet threshold in the reference year
        qualifying_stations = set(
            coverage.loc[
                (coverage['year'] == reference_year) & (coverage['sufficient_coverage']),
                station_col
            ].unique()
        )
    else:
        # If no reference year is provided, qualify stations that meet the threshold in ANY year
        qualifying_stations = set(
            coverage.loc[coverage['sufficient_coverage'], station_col].unique()
        )

    # --- Keep ALL rows for qualifying stations (across all years) ---
    out = df[df[station_col].isin(qualifying_stations)].copy()

    # Optionally attach coverage metrics (for all station-years) for traceability
    if keep_coverage_columns:
        out = out.merge(
            coverage,
            on=[station_col, 'year'],
            how='left',
            validate='many_to_one'
        )

    return out

def add_cams_daily_dust_by_station(
    cams_ds: xr.Dataset,
    df: pd.DataFrame,
    var_name: str = 'dust',
    time_col: str = 'day',
    lat_col: str = 'Latitude',
    lon_col: str = 'Longitude',
    station_col: str = 'Samplingpoint',
    spatial_method: str = 'linear'  # 'linear' (bilinear) or 'nearest'
) -> pd.DataFrame:
    """
    Add CAMS daily mean dust to df efficiently by looping over unique stations.

    For each station:
      - Interpolate CAMS daily mean field once at (lat, lon) to get a time series
      - Select nearest in time for all station rows in a single vectorized call

    Parameters
    ----------
    cams_ds : xr.Dataset
        CAMS dataset with dims [time, lat, lon] and variable `var_name` (e.g., 'dust').
    df : pd.DataFrame
        DataFrame with columns [time_col, lat_col, lon_col, station_col].
    var_name : str
        CAMS variable to sample (default 'dust').
    time_col : str
        DataFrame column name with daily timestamps (date/datetime; timezone OK).
    lat_col, lon_col : str
        Latitude & longitude column names in decimal degrees.
    station_col : str
        Station identifier column name.
    spatial_method : str
        'linear' (bilinear interpolation) or 'nearest' for lat/lon.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new column 'cams_dust'.
    """
    if var_name not in cams_ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in CAMS dataset.")
    daily_cams = cams_ds
    # Robust coord names
    lat_name = next((d for d in daily_cams.dims if d.lower().startswith('lat')), 'lat')
    lon_name = next((d for d in daily_cams.dims if d.lower().startswith('lon')), 'lon')

    # CAMS domain bounds (for clipping)
    lat_min = float(daily_cams[lat_name].min())
    lat_max = float(daily_cams[lat_name].max())
    lon_min = float(daily_cams[lon_name].min())
    lon_max = float(daily_cams[lon_name].max())

    out = df.copy()

    # Normalize times: midnight UTC, then drop tz to match xarray's naive time
    out[time_col] = (
        pd.to_datetime(out[time_col], utc=True, errors='coerce')
          .dt.floor('D')
          .dt.tz_convert(None)
    )

    # Ensure numeric lat/lon
    out[lat_col] = pd.to_numeric(out[lat_col], errors='coerce')
    out[lon_col] = pd.to_numeric(out[lon_col], errors='coerce')

    # Prepare output column
    out['cams_dust'] = np.nan

    # 2) Unique stations (first lat/lon per station)
    stations = (
        out[[station_col, lat_col, lon_col]]
        .dropna(subset=[station_col, lat_col, lon_col])
        .drop_duplicates(subset=[station_col], keep='first')
    )

    da = daily_cams[var_name]  # (time, lat, lon)

    # 3) Loop over stations, do one spatial interpolation & vectorized time selection
    for station_id, lat0, lon0 in stations.itertuples(index=False, name=None):
        # Clip to CAMS domain to avoid NaNs at edges
        lat0 = float(np.clip(lat0, lat_min, lat_max))
        lon0 = float(np.clip(lon0, lon_min, lon_max))

        # Interpolate the full daily time series at the station location (fast)
        ts_station = da.interp({lat_name: lat0, lon_name: lon0}, method=spatial_method)  # dims: time

        # All rows for this station
        mask = (out[station_col] == station_id)
        t_values = out.loc[mask, time_col].values

        # Vectorized nearest-time selection for those rows
        t_da = xr.DataArray(t_values, dims='index')  # aligns to row order
        vals = ts_station.sel(time=t_da, method='nearest').values  # dims: index

        # Optional: fallback to nearest spatial if any NaNs (e.g., outside convex hull)
        if np.isnan(vals).any():
            ts_station_nearest = da.interp({lat_name: lat0, lon_name: lon0}, method='nearest')
            vals = ts_station_nearest.sel(time=t_da, method='nearest').values

        out.loc[mask, 'cams_dust'] = np.asarray(vals, dtype=np.float32)

    return out


def compute_station_baseline(st_all: pd.DataFrame,
                             st_2024: pd.DataFrame,
                             time_col: str = 'day',
                             value_col: str = 'observed_PM10',
                             neighbor_n: int = 15) -> pd.Series:
    """
    For one station:
      - Find nondust days across ALL years (st_all)
      - For each dust & exceedance day in st_2024, take the last `neighbor_n` nondust days before
        and the next `neighbor_n` nondust days after -> median.
    Returns a Series aligned to st_2024.index with medians where applicable (NaN otherwise).
    """
    # Nondust pool across all years
    mask_nondust = (~st_all['dust_flag'].astype(bool)) & st_all[time_col].notna() & st_all[value_col].notna()
    nd_times = st_all.loc[mask_nondust, time_col].to_numpy()
    nd_vals  = st_all.loc[mask_nondust, value_col].to_numpy()

    # Sort nondust by time
    order = np.argsort(nd_times)
    nd_times = nd_times[order]
    nd_vals  = nd_vals[order]

    # Dust & exceedance days in 2024
    mask_dust_exc = st_2024['dust_flag'].astype(bool) & st_2024['Exceedance'].astype(bool)
    dust_idx_2024 = st_2024.index[mask_dust_exc]
    if dust_idx_2024.empty or nd_times.size == 0:
        return pd.Series(index=st_2024.index, dtype='float32')  # all NaN

    # Vector of target times
    t_targets = st_2024.loc[dust_idx_2024, time_col].to_numpy()

    # For each target time, find insertion position in nondust times
    pos = np.searchsorted(nd_times, t_targets, side='left')

    # Compute medians (loop over dust days of this station only; NumPy, not pandas)
    medians = np.full(t_targets.shape[0], np.nan, dtype='float32')
    for i, p in enumerate(pos):
        # Previous `neighbor_n` nondust days
        start_b = max(0, p - neighbor_n)
        before_vals = nd_vals[start_b:p]

        # Next `neighbor_n` nondust days
        end_a = min(nd_vals.shape[0], p + neighbor_n)
        after_vals = nd_vals[p:end_a]

        window = np.concatenate([before_vals, after_vals])
        if window.size > 0:
            medians[i] = np.nanmedian(window)  # robust to NaNs if any slipped in

    # Build result Series aligned to st_2024
    result = pd.Series(index=st_2024.index, dtype='float32')
    result.loc[dust_idx_2024] = medians
    return result

########################################################
############ daily data processing function ############
########################################################
#--- for EEA daily data filter functions
def calculate_data_coverage(df, 
                            start : str,
                            end : str,
                            min_pct: int =75, 
                            date_col : str = 'Start'
 ):
    """
    Coverage per station across the chosen period [start, end] (inclusive).
    Automatically handles leap years and partial ranges.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['tmpdate'] = df[date_col].dt.normalize()

    period_start = pd.to_datetime(start) if start is not None else df['tmpdate'].min()
    period_end   = pd.to_datetime(end)   if end   is not None else df['tmpdate'].max()
    period_end   = period_end.normalize()

    # Expected days: inclusive date range count
    expected_days_total = pd.date_range(period_start, period_end, freq='D').size

    # Observed unique days per station within window
    observed = (
        df[(df['tmpdate'] >= period_start) & (df['tmpdate'] <= period_end)]
          .groupby('Samplingpoint')['tmpdate']
          .nunique()
          .rename('unique_days')
          .to_frame()
    )

    observed['expected_days'] = expected_days_total
    observed['coverage_percentage'] = (observed['unique_days'] / observed['expected_days']) * 100.0
    observed['sufficient_coverage'] = observed['coverage_percentage'] >= float(min_pct)

    return observed

#--- for EEA hourly data filter functions
def filter_daily_by_coverage(
    df_daily: pd.DataFrame,
    day_col: str = 'day',
    station_col: str = 'Samplingpoint',
    reference_year: int | None = None,
    min_pct: float = 75.0,
    keep_coverage_columns: bool = True
) -> pd.DataFrame:
    """
    Keep ALL rows for stations that have coverage >= min_pct IN the reference year.
    Example: if a station has >=75% coverage in 2024, keep its 2023 data as well since average window need longer rolling time.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily dataframe (one row per day per station).
    day_col : str
        Column with daily timestamp (date or datetime).
    station_col : str
        Station/sampling point identifier column.
    reference_year : int | None
        The year used to evaluate coverage. If None, coverage is evaluated
        across ALL years (station-level over the whole dataset).
    min_pct : float
        Coverage threshold (default 75.0).
    keep_coverage_columns : bool
        If True, merge per-(station, year) coverage metrics onto the output.

    Returns
    -------
    pd.DataFrame
        All rows for stations that meet the coverage criterion in the reference year.
        If keep_coverage_columns=True, includes coverage columns:
        ['year','unique_days','total_days','coverage_percentage','sufficient_coverage'].
    """
    df = df_daily.copy()

    # Normalize day and extract year
    df[day_col] = pd.to_datetime(df[day_col], utc=True, errors='coerce').dt.floor('D')
    df = df.dropna(subset=[day_col, station_col])
    df['year'] = df[day_col].dt.year

    # --- Compute coverage per (station, year) ---
    coverage = (
        df.groupby([station_col, 'year'])[day_col]
          .nunique()
          .rename('unique_days')
          .reset_index()
    )
    coverage['total_days'] = coverage['year'].apply(lambda y: 366 if calendar.isleap(y) else 365)
    coverage['coverage_percentage'] = (coverage['unique_days'] / coverage['total_days']) * 100.0
    coverage['sufficient_coverage'] = coverage['coverage_percentage'] >= min_pct

    # --- Decide which stations qualify ---
    if reference_year is not None:
        # Stations that meet threshold in the reference year
        qualifying_stations = set(
            coverage.loc[
                (coverage['year'] == reference_year) & (coverage['sufficient_coverage']),
                station_col
            ].unique()
        )
    else:
        # If no reference year is provided, qualify stations that meet the threshold in ANY year
        qualifying_stations = set(
            coverage.loc[coverage['sufficient_coverage'], station_col].unique()
        )

    # --- Keep ALL rows for qualifying stations (across all years) ---
    out = df[df[station_col].isin(qualifying_stations)].copy()

    # Optionally attach coverage metrics (for all station-years) for traceability
    if keep_coverage_columns:
        out = out.merge(
            coverage,
            on=[station_col, 'year'],
            how='left',
            validate='many_to_one'
        )

    return out

# --- for EEA hourly data to interpolate 

def add_cams_daily_dust_by_station(
    cams_ds: xr.Dataset,
    df: pd.DataFrame,
    var_name: str = 'dust',
    time_col: str = 'day',
    lat_col: str = 'Latitude',
    lon_col: str = 'Longitude',
    station_col: str = 'Samplingpoint',
    spatial_method: str = 'linear'  # 'linear' (bilinear) or 'nearest'
) -> pd.DataFrame:
    """
    Add CAMS daily mean dust to df efficiently by looping over unique stations.

    For each station:
      - Interpolate CAMS daily mean field once at (lat, lon) to get a time series
      - Select nearest in time for all station rows in a single vectorized call

    Parameters
    ----------
    cams_ds : xr.Dataset
        CAMS dataset with dims [time, lat, lon] and variable `var_name` (e.g., 'dust').
    df : pd.DataFrame
        DataFrame with columns [time_col, lat_col, lon_col, station_col].
    var_name : str
        CAMS variable to sample (default 'dust').
    time_col : str
        DataFrame column name with daily timestamps (date/datetime; timezone OK).
    lat_col, lon_col : str
        Latitude & longitude column names in decimal degrees.
    station_col : str
        Station identifier column name.
    spatial_method : str
        'linear' (bilinear interpolation) or 'nearest' for lat/lon.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new column 'cams_dust'.
    """
    if var_name not in cams_ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in CAMS dataset.")
    daily_cams = cams_ds
    # Robust coord names
    lat_name = next((d for d in daily_cams.dims if d.lower().startswith('lat')), 'lat')
    lon_name = next((d for d in daily_cams.dims if d.lower().startswith('lon')), 'lon')

    # CAMS domain bounds (for clipping)
    lat_min = float(daily_cams[lat_name].min())
    lat_max = float(daily_cams[lat_name].max())
    lon_min = float(daily_cams[lon_name].min())
    lon_max = float(daily_cams[lon_name].max())

    out = df.copy()

    # Normalize times: midnight UTC, then drop tz to match xarray's naive time
    out[time_col] = (
        pd.to_datetime(out[time_col], utc=True, errors='coerce')
          .dt.floor('D')
          .dt.tz_convert(None)
    )

    # Ensure numeric lat/lon
    out[lat_col] = pd.to_numeric(out[lat_col], errors='coerce')
    out[lon_col] = pd.to_numeric(out[lon_col], errors='coerce')

    # Prepare output column
    out_name = f'cams_{var_name}'
    out[out_name] = np.nan

    # 2) Unique stations (first lat/lon per station)
    stations = (
        out[[station_col, lat_col, lon_col]]
        .dropna(subset=[station_col, lat_col, lon_col])
        .drop_duplicates(subset=[station_col], keep='first')
    )

    da = daily_cams[var_name]  # (time, lat, lon)

    # 3) Loop over stations, do one spatial interpolation & vectorized time selection
    for station_id, lat0, lon0 in stations.itertuples(index=False, name=None):
        # Clip to CAMS domain to avoid NaNs at edges
        lat0 = float(np.clip(lat0, lat_min, lat_max))
        lon0 = float(np.clip(lon0, lon_min, lon_max))

        # Interpolate the full daily time series at the station location (fast)
        ts_station = da.interp({lat_name: lat0, lon_name: lon0}, method=spatial_method)  # dims: time

        # All rows for this station
        mask = (out[station_col] == station_id)
        t_values = out.loc[mask, time_col].values

        # Vectorized nearest-time selection for those rows
        t_da = xr.DataArray(t_values, dims='index')  # aligns to row order
        vals = ts_station.sel(time=t_da, method='nearest').values  # dims: index

        # Optional: fallback to nearest spatial if any NaNs (e.g., outside convex hull)
        if np.isnan(vals).any():
            ts_station_nearest = da.interp({lat_name: lat0, lon_name: lon0}, method='nearest')
            vals = ts_station_nearest.sel(time=t_da, method='nearest').values

        out.loc[mask, out_name] = np.asarray(vals, dtype=np.float32)

    return out

#------- daily data median-------#
def compute_median_for_station(station_df: pd.DataFrame,
                               flag_col : str = 'dust_flag',
                               obs_col  : str = 'observed_PM10',
                               date_col : str = 'Start',
                               Exceedance_col : str='Exceedance'
                               ):
    # Sort by date
    station_df = station_df.sort_values('Start')

    # Separate nondust days
    nondust = station_df[station_df[flag_col] == False][[date_col,obs_col]].copy()
    nd_dates = nondust[date_col].to_numpy()
    nd_vals = nondust[obs_col].to_numpy(dtype=float)

    # Prepare result
    result = pd.Series(index=station_df.index, dtype=float)

    # Identify dust days with Exceedance
    dust_days = station_df[(station_df[flag_col]) & (station_df[Exceedance_col])]

    if nondust.empty or dust_days.empty:
        return result

    # Vectorized search for positions
    positions = np.searchsorted(nd_dates, dust_days[date_col].to_numpy())

    for i, idx in enumerate(dust_days.index):
        pos = positions[i]
        before_slice = nd_vals[max(0, pos-15):pos]
        after_slice = nd_vals[pos:pos+15]
        window = np.concatenate([before_slice, after_slice])
        if window.size > 0:
            result[idx] = np.median(window)

    return result
