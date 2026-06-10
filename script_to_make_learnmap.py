#!/usr/bin/env python3
import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from colours import *
from input_files import *
from helpers import *
from trainstation_types import *


def get_intercities(url):
    df = pd.read_csv(url)
    df = df[["name_long", "type"]]
    intercities = df[df["type"].isin(intercity_types)]
    return intercities["name_long"]


def get_sprinters(url):
    df = pd.read_csv(url)
    df = df[["name_long", "type"]]
    sprinters = df[df["type"].isin(sprinter_types)]
    return sprinters["name_long"]


def make_file(output_file):
    # ── 1. Fetch data ──────────────────────────────────────────────────────────
    print("Fetching data …")
    gdf = gpd.read_file(INPUT_FILE)
    lines_gdf = gdf[gdf.geometry.geom_type.isin({"LineString", "MultiLineString"})]

    # ── 2. Filter to main railway types ───────────────────────────────────────
    MAIN_TYPES = {"rail"}
    lines_main = lines_gdf[lines_gdf["railway"].isin(MAIN_TYPES)].copy()
    lines_other = lines_gdf[~lines_gdf["railway"].isin(MAIN_TYPES)].copy()

    # Stations / halts only
    stations_gdf = gpd.read_file(INPUT_OV_FILE)

    STATION_TYPES = {"Treinstation"}
    stations = stations_gdf[stations_gdf["Type_halte"].isin(STATION_TYPES)].copy()
    stations = stations.drop_duplicates(subset="Naam", keep="first")
    stations = stations.dropna(subset=["geometry"])

    # Give 25 random stations a sequential number

    stations_2 = stations.copy()
    stations_2["Naam"] = stations_2["Naam"].apply(lambda x: split_name(x))
    stations_2 = stations_2.reset_index(drop=True)
    stations_2["map_number"] = stations_2.index + 1

    intercity_stations = stations_2[stations_2["Naam"].isin(get_intercities(INTERCITY))]

    sprinter_stations = stations_2[stations_2["Naam"].isin(get_sprinters(INTERCITY))]

    print(
        f"  {len(lines_main)} main-line segments, {len(intercity_stations)} IC stations, {len(sprinter_stations)} SPR stations"
    )
    provinces = load_admin_borders(GADM_URL)
    country = provinces.dissolve()  # union of all provinces = country outline

    # ── 3. Reproject to Dutch RD New (EPSG:28992) ─────────────────────────────
    lines_main = lines_main.to_crs(epsg=28992)
    stations = stations.to_crs(epsg=28992)

    # ── 4. Build the figure ───────────────────────────────────────────────────

    fig = plt.figure(figsize=(21, 30), facecolor=PARCHMENT)

    # Map area (left ~70 % of width)
    ax_map = fig.add_axes([0.02, 0.06, 0.66, 0.88], facecolor=PARCHMENT)

    # Legend area (right panel)
    ax_leg = fig.add_axes([0.70, 0.06, 0.28, 0.88], facecolor=PARCHMENT)
    ax_leg.axis("off")

    # ── 4a. Decorative graticule ───────────────────────────────────────────────
    xmin, ymin, xmax, ymax = lines_main.total_bounds
    pad_x = (xmax - xmin) * 0.04
    pad_y = (ymax - ymin) * 0.04
    ax_map.set_xlim(xmin - pad_x, xmax + pad_x)
    ax_map.set_ylim(ymin - pad_y, ymax + pad_y)

    for x in np.linspace(xmin - pad_x, xmax + pad_x, 7):
        ax_map.axvline(x, color=GRID_COLOR, lw=0.4, zorder=0, alpha=0.7)
    for y in np.linspace(ymin - pad_y, ymax + pad_y, 9):
        ax_map.axhline(y, color=GRID_COLOR, lw=0.4, zorder=0, alpha=0.7)

    # ── Province borders ──────────────────────────────────────────────────────────
    provinces.plot(
        ax=ax_map,
        facecolor="none",
        edgecolor=BROWN,
        linewidth=0.6,
        linestyle="--",
        zorder=1,
    )

    # ── Country outline ───────────────────────────────────────────────────────────
    country.plot(
        ax=ax_map,
        facecolor="none",
        edgecolor=INK,
        linewidth=1.8,
        zorder=1,
    )

    # ── 4b. Draw railway lines ─────────────────────────────────────────────────
    def plot_lines(gdf, color, lw, zorder, label, ls="-"):
        for geom in gdf.geometry:
            if geom is None:
                continue
            if geom.geom_type == "LineString":
                xs, ys = geom.xy
                ax_map.plot(
                    xs,
                    ys,
                    color=color,
                    lw=lw,
                    ls=ls,
                    solid_capstyle="round",
                    zorder=zorder,
                )
            elif geom.geom_type == "MultiLineString":
                for part in geom.geoms:
                    xs, ys = part.xy
                    ax_map.plot(
                        xs,
                        ys,
                        color=color,
                        lw=lw,
                        ls=ls,
                        solid_capstyle="round",
                        zorder=zorder,
                    )

    # Shadow / halo for the main lines (gives a printed-map look)
    plot_lines(lines_main, "#C4A882", lw=3.2, zorder=2, label=None)
    plot_lines(lines_main, RAIL_MAIN, lw=1.6, zorder=3, label="Railway")

    # ── 4c. Draw stations ─────────────────────────────────────────────────────
    for _, row in stations.iterrows():
        x, y = row.geometry.x, row.geometry.y

        # Dot
        ax_map.plot(
            x,
            y,
            "o",
            color=PARCHMENT,
            markersize=3,
            markeredgecolor=OTHERSTATIONS,
            markeredgewidth=1,
            zorder=4,
        )

    for _, row in intercity_stations.iterrows():
        x, y = row.geometry.x, row.geometry.y
        num = row["map_number"]

        # Dot
        ax_map.plot(
            x,
            y,
            "o",
            color=BLUE,
            markersize=7,
            markeredgecolor=BLUE,
            markeredgewidth=1.2,
            zorder=5,
        )

        # Number label with halo
        txt = ax_map.text(
            x,
            y,
            str(num),
            ha="center",
            va="center",
            fontsize=4.5,
            fontweight="bold",
            color=BLUE,
            zorder=6,
        )
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground=PARCHMENT)])
    for _, row in sprinter_stations.iterrows():
        x, y = row.geometry.x, row.geometry.y
        num = row["map_number"]

        # Dot
        ax_map.plot(
            x,
            y,
            "o",
            color=GREEN,
            markersize=5,
            markeredgecolor=GREEN,
            markeredgewidth=1.1,
            zorder=4,
        )

        # Number label with halo
        txt = ax_map.text(
            x,
            y,
            str(num),
            ha="center",
            va="center",
            fontsize=4.5,
            fontweight="bold",
            color=GREEN,
            zorder=6,
        )
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground=PARCHMENT)])

    # ── 4d. Map decoration ────────────────────────────────────────────────────
    ax_map.set_aspect("equal")
    ax_map.tick_params(colors=INK, labelsize=6)
    ax_map.spines[:].set_edgecolor(INK)
    ax_map.spines[:].set_linewidth(1.2)
    for spine in ax_map.spines.values():
        spine.set_visible(True)

    # Compass rose (simple N arrow)
    ar_x = xmax + pad_x * 0.5
    ar_y = ymin + (ymax - ymin) * 0.08
    ax_map.annotate(
        "",
        xy=(ar_x, ar_y + (ymax - ymin) * 0.04),
        xytext=(ar_x, ar_y),
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2),
        zorder=7,
    )
    ax_map.text(
        ar_x,
        ar_y + (ymax - ymin) * 0.046,
        "N",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=INK,
        zorder=7,
    )

    # ── 4e. Title ─────────────────────────────────────────────────────────────
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

    # ── 5. Legend panel ───────────────────────────────────────────────────────
    COLS = 2
    col_w = 0.5  # fraction of legend width per column
    row_h = 0.018  # fraction of figure height per row

    ax_leg.text(
        0.5,
        0.97,
        "STATIONSLIJST",
        ha="center",
        va="top",
        transform=ax_leg.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        fontfamily="garamond",
    )
    ax_leg.text(
        0.5,
        0.955,
        "Index of Stations",
        ha="center",
        va="top",
        transform=ax_leg.transAxes,
        fontsize=7,
        color=INK,
        fontstyle="italic",
        fontfamily="garamond",
    )

    # Divider line
    ax_leg.axhline(0.945, color=INK, lw=0.8)

    # Sort stations for the legend
    legend_stations = stations_2.copy()
    legend_stations["name_clean"] = (
        legend_stations["Naam"]
        .fillna(legend_stations.get("name:en", ""))
        .fillna("(unnamed)")
        .astype(str)
        .str.strip()
        .replace("", "(unnamed)")
    )

    legend_stations["name_cleaner"] = legend_stations["name_clean"].apply(
        lambda x: split_name(x)
    )
    legend_stations = legend_stations.sort_values("map_number")

    n = len(legend_stations)
    rows_per_col = (n + COLS - 1) // COLS
    font_size = max(4.5, min(6.5, 6.5 - (rows_per_col - 40) * 0.04))

    for i, (_, row) in enumerate(legend_stations.iterrows()):
        col = i // rows_per_col
        r = i % rows_per_col
        x = 0.02 + col * col_w
        y = 0.935 - r * row_h * (0.935 / (rows_per_col * row_h + 0.01))

        ax_leg.text(
            x,
            y,
            f"{row['map_number']:>3}. {row['name_cleaner']}",
            transform=ax_leg.transAxes,
            fontsize=font_size,
            color=INK,
            fontfamily="monospace",
            va="top",
        )

    # Map-symbol legend
    legend_y = 0.04
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

    # Border around legend panel
    for spine in ax_leg.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(INK)
        spine.set_linewidth(1)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    print(f"Saving {output_file} …")
    fig.savefig(
        output_file,
        format="jpeg",
        dpi=200,
        bbox_inches="tight",
        pil_kwargs={"quality": 92},
    )
    plt.close(fig)
    print(f"Done → {os.path.abspath(output_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_file", required=True, action="store")
    args = parser.parse_args()
    output_file = args.output_file
    if not output_file.endswith(".jpeg") and not output_file.endswith(".jpg"):
        output_file = output_file + ".jpg"
    make_file(output_file)
