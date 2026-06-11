#!/usr/bin/env python3
import argparse
import os
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from colours import (
    BLUE,
    BROWN,
    GREEN,
    GRID_COLOR,
    INK,
    OTHERSTATIONS,
    PARCHMENT,
    RAIL_MAIN,
)
from helpers import load_admin_borders, split_name
from input_files import GADM_URL, INPUT_FILE, INPUT_OV_FILE, INTERCITY
from trainstation_types import intercity_types, sprinter_types

RD_NEW_EPSG = 28992

MAIN_RAILWAY_TYPES = {"rail"}
STATION_TYPES = {"Treinstation"}

FIGSIZE = (21, 30)
MAP_AX_BOUNDS = [0.02, 0.06, 0.66, 0.88]
LEGEND_AX_BOUNDS = [0.70, 0.06, 0.28, 0.88]

LEGEND_COLUMNS = 2
LEGEND_COLUMN_WIDTH = 0.5
LEGEND_ROW_HEIGHT = 0.018


def get_station_names_by_types(csv_file: str, station_types: set[str]) -> pd.Series:
    """Return station names matching the given train station types."""
    stations = pd.read_csv(csv_file)[["name_long", "type"]]
    return stations[stations["type"].isin(station_types)]["name_long"]


def get_intercities(csv_file: str) -> pd.Series:
    """Return intercity station names."""
    return get_station_names_by_types(csv_file, intercity_types)


def get_sprinters(csv_file: str) -> pd.Series:
    """Return sprinter station names."""
    return get_station_names_by_types(csv_file, sprinter_types)


def load_input_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load railway lines, train stations, and provinces."""
    print("Fetching data …")

    railway_data = gpd.read_file(INPUT_FILE)
    station_data = gpd.read_file(INPUT_OV_FILE)
    provinces = load_admin_borders(GADM_URL)

    railway_lines = railway_data[
        railway_data.geometry.geom_type.isin({"LineString", "MultiLineString"})
    ]
    main_lines = railway_lines[railway_lines["railway"].isin(MAIN_RAILWAY_TYPES)].copy()

    stations = station_data[station_data["Type_halte"].isin(STATION_TYPES)].copy()
    stations = stations.drop_duplicates(subset="Naam", keep="first")
    stations = stations.dropna(subset=["geometry"])
    stations["Naam"] = stations["Naam"].apply(split_name)

    return main_lines, stations, provinces


def number_stations(stations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a sequential map number to each station."""
    numbered_stations = stations.copy()
    numbered_stations = numbered_stations.reset_index(drop=True)
    numbered_stations["map_number"] = numbered_stations.index + 1
    return numbered_stations


def classify_stations(
    stations: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split numbered stations into intercity and sprinter station subsets."""
    intercity_names = get_intercities(INTERCITY)
    sprinter_names = get_sprinters(INTERCITY)

    intercity_stations = stations[stations["Naam"].isin(intercity_names)]
    sprinter_stations = stations[stations["Naam"].isin(sprinter_names)]

    return intercity_stations, sprinter_stations


def configure_map_bounds(
    ax_map,
    lines: gpd.GeoDataFrame,
) -> tuple[float, float, float, float, float, float]:
    """Set map bounds and return bounds plus padding."""
    xmin, ymin, xmax, ymax = lines.total_bounds
    pad_x = (xmax - xmin) * 0.04
    pad_y = (ymax - ymin) * 0.04

    ax_map.set_xlim(xmin - pad_x, xmax + pad_x)
    ax_map.set_ylim(ymin - pad_y, ymax + pad_y)

    return xmin, ymin, xmax, ymax, pad_x, pad_y


def draw_grid(
    ax_map,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    pad_x: float,
    pad_y: float,
) -> None:
    """Draw the decorative map grid."""
    for x in np.linspace(xmin - pad_x, xmax + pad_x, 7):
        ax_map.axvline(x, color=GRID_COLOR, lw=0.4, zorder=0, alpha=0.7)

    for y in np.linspace(ymin - pad_y, ymax + pad_y, 9):
        ax_map.axhline(y, color=GRID_COLOR, lw=0.4, zorder=0, alpha=0.7)


def draw_railway_lines(
    ax_map,
    lines: gpd.GeoDataFrame,
    color: str,
    line_width: float,
    zorder: int,
    linestyle: str = "-",
) -> None:
    """Draw LineString and MultiLineString railway geometries."""
    for geom in lines.geometry:
        if geom is None:
            continue

        geometries = geom.geoms if geom.geom_type == "MultiLineString" else [geom]

        for line in geometries:
            xs, ys = line.xy
            ax_map.plot(
                xs,
                ys,
                color=color,
                lw=line_width,
                ls=linestyle,
                solid_capstyle="round",
                zorder=zorder,
            )


def draw_base_stations(ax_map, stations: gpd.GeoDataFrame) -> None:
    """Draw all stations as small background dots."""
    for _, row in stations.iterrows():
        ax_map.plot(
            row.geometry.x,
            row.geometry.y,
            "o",
            color=PARCHMENT,
            markersize=3,
            markeredgecolor=OTHERSTATIONS,
            markeredgewidth=1,
            zorder=4,
        )


def draw_numbered_stations(
    ax_map,
    stations: gpd.GeoDataFrame,
    color: str,
    markersize: float,
    markeredgewidth: float,
    zorder: int,
) -> None:
    """Draw numbered stations in the requested style."""
    for _, row in stations.iterrows():
        x, y = row.geometry.x, row.geometry.y

        ax_map.plot(
            x,
            y,
            "o",
            color=color,
            markersize=markersize,
            markeredgecolor=color,
            markeredgewidth=markeredgewidth,
            zorder=zorder,
        )

        label = ax_map.text(
            x,
            y,
            str(row["map_number"]),
            ha="center",
            va="center",
            fontsize=4.5,
            fontweight="bold",
            color=color,
            zorder=6,
        )
        label.set_path_effects([pe.withStroke(linewidth=1.5, foreground=PARCHMENT)])


def style_map_axes(ax_map) -> None:
    """Apply final styling to the map axes."""
    ax_map.set_aspect("equal")
    ax_map.tick_params(colors=INK, labelsize=6)
    ax_map.spines[:].set_edgecolor(INK)
    ax_map.spines[:].set_linewidth(1.2)

    for spine in ax_map.spines.values():
        spine.set_visible(True)


def draw_compass(
    ax_map,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    pad_x: float,
) -> None:
    """Draw a simple north arrow."""
    arrow_x = xmax + pad_x * 0.5
    arrow_y = ymin + (ymax - ymin) * 0.08

    ax_map.annotate(
        "",
        xy=(arrow_x, arrow_y + (ymax - ymin) * 0.04),
        xytext=(arrow_x, arrow_y),
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.2},
        zorder=7,
    )

    ax_map.text(
        arrow_x,
        arrow_y + (ymax - ymin) * 0.046,
        "N",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=INK,
        zorder=7,
    )


def add_title(fig) -> None:
    """Add the map title."""
    fig.text(
        0.35,
        0.90,
        "SPOORWEGEN DER NEDERLANDEN",
        ha="center",
        va="top",
        fontsize=55,
        fontweight="bold",
        color=INK,
        fontfamily="garamond",
    )

    fig.text(
        0.35,
        0.845,
        "Dutch Railway Network",
        ha="center",
        va="top",
        fontsize=30,
        color=INK,
        fontstyle="italic",
        fontfamily="garamond",
    )


def build_station_legend(ax_legend, stations: gpd.GeoDataFrame) -> None:
    """Build the station index legend."""
    ax_legend.axis("off")

    ax_legend.text(
        0.5,
        0.97,
        "STATIONSLIJST",
        ha="center",
        va="top",
        transform=ax_legend.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        fontfamily="garamond",
    )

    ax_legend.text(
        0.5,
        0.955,
        "Index of Stations",
        ha="center",
        va="top",
        transform=ax_legend.transAxes,
        fontsize=7,
        color=INK,
        fontstyle="italic",
        fontfamily="garamond",
    )

    ax_legend.axhline(0.945, color=INK, lw=0.8)

    legend_stations = stations.copy()
    legend_stations["name_clean"] = (
        legend_stations["Naam"]
        .fillna(legend_stations.get("name:en", ""))
        .fillna("(unnamed)")
        .astype(str)
        .str.strip()
        .replace("", "(unnamed)")
    )
    legend_stations["name_cleaner"] = legend_stations["name_clean"].apply(split_name)
    legend_stations = legend_stations.sort_values("map_number")

    station_count = len(legend_stations)
    rows_per_column = (station_count + LEGEND_COLUMNS - 1) // LEGEND_COLUMNS
    font_size = max(4.5, min(6.5, 6.5 - (rows_per_column - 40) * 0.04))

    for index, (_, row) in enumerate(legend_stations.iterrows()):
        column = index // rows_per_column
        row_number = index % rows_per_column

        x = 0.02 + column * LEGEND_COLUMN_WIDTH
        y = 0.935 - row_number * LEGEND_ROW_HEIGHT * (
            0.935 / (rows_per_column * LEGEND_ROW_HEIGHT + 0.01)
        )

        ax_legend.text(
            x,
            y,
            f"{row['map_number']:>3}. {row['name_cleaner']}",
            transform=ax_legend.transAxes,
            fontsize=font_size,
            color=INK,
            fontfamily="monospace",
            va="top",
        )

    for spine in ax_legend.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(INK)
        spine.set_linewidth(1)


def add_map_symbol_legend(ax_map) -> None:
    """Add the map-symbol legend."""
    legend_elements = [
        Line2D([0], [0], color=RAIL_MAIN, lw=2, label="Rail"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PARCHMENT,
            markeredgecolor=GREEN,
            markersize=7,
            label="Sprinter Station",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PARCHMENT,
            markeredgecolor=BLUE,
            markersize=7,
            label="Intercity Station",
        ),
    ]

    ax_map.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
        framealpha=0.85,
        edgecolor=INK,
        facecolor=PARCHMENT,
        fontsize=7,
        ncol=1,
    )


def save_figure(fig, output_file: str) -> None:
    """Save the figure as a JPEG."""
    print(f"Saving {output_file} …")
    fig.savefig(
        output_file,
        format="jpeg",
        dpi=200,
        bbox_inches="tight",
        pil_kwargs={"quality": 92},
    )
    print(f"Done → {os.path.abspath(output_file)}")


def make_file(output_file: str) -> None:
    """Create the learning map."""
    main_lines, stations, provinces = load_input_data()

    numbered_stations = number_stations(stations)
    intercity_stations, sprinter_stations = classify_stations(numbered_stations)

    print(
        f"  {len(main_lines)} main-line segments, "
        f"{len(intercity_stations)} IC stations, "
        f"{len(sprinter_stations)} SPR stations"
    )

    country = provinces.dissolve()

    main_lines = main_lines.to_crs(epsg=RD_NEW_EPSG)
    stations = stations.to_crs(epsg=RD_NEW_EPSG)
    numbered_stations = numbered_stations.to_crs(epsg=RD_NEW_EPSG)
    intercity_stations = intercity_stations.to_crs(epsg=RD_NEW_EPSG)
    sprinter_stations = sprinter_stations.to_crs(epsg=RD_NEW_EPSG)

    fig = plt.figure(figsize=FIGSIZE, facecolor=PARCHMENT)
    ax_map = fig.add_axes(MAP_AX_BOUNDS, facecolor=PARCHMENT)
    ax_legend = fig.add_axes(LEGEND_AX_BOUNDS, facecolor=PARCHMENT)

    xmin, ymin, xmax, ymax, pad_x, pad_y = configure_map_bounds(ax_map, main_lines)
    draw_grid(ax_map, xmin, ymin, xmax, ymax, pad_x, pad_y)

    provinces.plot(
        ax=ax_map,
        facecolor="none",
        edgecolor=BROWN,
        linewidth=0.6,
        linestyle="--",
        zorder=1,
    )

    country.plot(
        ax=ax_map,
        facecolor="none",
        edgecolor=INK,
        linewidth=1.8,
        zorder=1,
    )

    draw_railway_lines(ax_map, main_lines, "#C4A882", line_width=3.2, zorder=2)
    draw_railway_lines(ax_map, main_lines, RAIL_MAIN, line_width=1.6, zorder=3)

    draw_base_stations(ax_map, stations)
    draw_numbered_stations(
        ax_map,
        intercity_stations,
        color=BLUE,
        markersize=7,
        markeredgewidth=1.2,
        zorder=5,
    )
    draw_numbered_stations(
        ax_map,
        sprinter_stations,
        color=GREEN,
        markersize=5,
        markeredgewidth=1.1,
        zorder=4,
    )

    style_map_axes(ax_map)
    draw_compass(ax_map, xmin, ymin, xmax, ymax, pad_x)
    add_title(fig)
    build_station_legend(ax_legend, numbered_stations)
    add_map_symbol_legend(ax_map)

    save_figure(fig, output_file)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_file", required=True)
    return parser.parse_args()


def normalize_output_file(output_file: str) -> str:
    """Ensure the output file has a JPEG extension."""
    if output_file.endswith((".jpeg", ".jpg")):
        return output_file

    return f"{output_file}.jpg"


def main() -> None:
    args = parse_args()
    make_file(normalize_output_file(args.output_file))


if __name__ == "__main__":
    main()
