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


def get_sample(url, ic, sp):
    df = pd.read_csv(url)
    df = df[["name_long", "type"]]
    intercities = df[df["type"].isin(intercity_types)]

    intercites_sample = intercities.sample(ic)
    sprinters = df[df["type"].isin(sprinter_types)]
    sprinters_sample = sprinters.sample(sp)

    df_totaal = pd.concat([intercites_sample, sprinters_sample])
    return df_totaal["name_long"]


def make_files(
    versie,
    output_file,
    output_file_legend,
    lines_main,
    stations,
    ostations,
    provinces,
    ic_sample_size,
    spr_sample_size,
):
    print(f"making version {versie}")
    # Give (ic_sample_size+spr_sample_size) random stations a sequential number

    overhoor_stations = ostations[
        ostations["Naam"].isin(
            get_sample(INTERCITY, ic=ic_sample_size, sp=spr_sample_size)
        )
    ]
    while len(overhoor_stations) < (ic_sample_size + spr_sample_size):
        overhoor_stations = pd.concat(
            [
                overhoor_stations,
                ostations[
                    ostations["Naam"].isin(
                        get_sample(INTERCITY, ic=ic_sample_size, sp=spr_sample_size)
                    )
                ],
            ]
        )
        overhoor_stations = overhoor_stations.drop_duplicates(
            subset="Naam", keep="first"
        )
        if len(overhoor_stations) > (ic_sample_size + spr_sample_size):
            overhoor_stations = overhoor_stations.head(ic_sample_size + spr_sample_size)

    overhoor_stations = overhoor_stations.reset_index(drop=True)
    overhoor_stations["map_number"] = overhoor_stations.index + 1
    print(f" {len(stations)} stations, {len(overhoor_stations)} overhoorstations")
    country = provinces.dissolve()  # union of all provinces = country outline

    # ── 3. Reproject to Dutch RD New (EPSG:28992) ─────────────────────────────
    lines_main = lines_main.to_crs(epsg=28992)
    stations = stations.to_crs(epsg=28992)

    # ── 4. Build the figure ───────────────────────────────────────────────────
    # Vintage palette
    fig = plt.figure(figsize=(21, 30), facecolor=WHITE)
    figL = plt.figure(figsize=(4.2, 6), facecolor=WHITE)

    ax_map = fig.add_axes([0.02, 0.06, 0.94, 0.94], facecolor=WHITE)

    # Legend area (right panel)
    ax_leg = figL.add_axes([0.02, 0.06, 0.88, 0.88], facecolor=WHITE)
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
        edgecolor=BROWN,  # warm brown, matches the parchment palette
        linewidth=0.6,
        linestyle="--",
        zorder=1,
    )

    # ── Country outline ───────────────────────────────────────────────────────────
    country.plot(
        ax=ax_map,
        facecolor="none",
        edgecolor=INK,  # dark ink, strong outer border
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
    plot_lines(lines_main, "#C4A882", lw=3, zorder=2, label=None)
    plot_lines(lines_main, RAIL_MAIN, lw=1.5, zorder=3, label="Railway")

    # ── 4c. Draw stations ─────────────────────────────────────────────────────
    for _, row in overhoor_stations.iterrows():
        # plot the not questioned stations AFTER the questioned stations, to get the map less confusing when overlapping
        x, y = row.geometry.x, row.geometry.y
        num = row["map_number"]

        # Dot
        ax_map.plot(
            x,
            y,
            "o",
            color=PARCHMENT,
            markersize=18,
            markeredgecolor=ACCENT,
            markeredgewidth=1.3,
            zorder=5,
        )

        # Number label with halo
        txt = ax_map.text(
            x,
            y,
            str(num),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=ACCENT,
            zorder=6,
        )
        txt.set_path_effects([pe.withStroke(linewidth=1, foreground=PARCHMENT)])

    for _, row in stations.iterrows():
        if row["Naam"] in list(overhoor_stations["Naam"]):
            continue
        x, y = row.geometry.x, row.geometry.y

        # Dot
        ax_map.plot(
            x,
            y,
            "o",
            color=PARCHMENT,
            markersize=5,
            markeredgecolor=OTHERSTATIONS,
            markeredgewidth=1,
            zorder=7,
        )

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
        0.50,
        0.97,
        "SPOORWEGEN DER NEDERLANDEN",
        ha="center",
        va="top",
        fontsize=26,
        fontweight="bold",
        color=INK,
        fontfamily="garamond",
    )
    fig.text(
        0.50,
        0.945,
        f"EXAM VERSION: {versie}",
        ha="center",
        va="top",
        fontsize=20,
        color=INK,
        fontfamily="garamond",
    )

    # ── 5. Legend panel ───────────────────────────────────────────────────────
    COLS = 2
    col_w = 0.5  # fraction of legend width per column
    row_h = 0.018  # fraction of figure height per row

    ax_leg.text(
        0.5,
        0.97,
        f"ANSWERS VERSION: {versie}",
        ha="center",
        va="top",
        transform=ax_leg.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        fontfamily="serif",
    )

    # Sort stations for the legend
    legend_stations = overhoor_stations.copy()
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
            markerfacecolor=WHITE,
            markeredgecolor=ACCENT,
            markersize=7,
            label="Station",
        ),
    ]
    ax_map.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
        framealpha=0.85,
        edgecolor=INK,
        facecolor=WHITE,
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
    print(f"Saving {output_file_legend} …")
    figL.savefig(
        output_file_legend,
        format="jpeg",
        dpi=200,
        bbox_inches="tight",
        pil_kwargs={"quality": 92},
    )
    plt.close(fig)
    print(f"Done → {os.path.abspath(output_file_legend)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", required=True, action="store")
    parser.add_argument(
        "--versions", required=False, action="store", type=int, default=3
    )
    parser.add_argument("--ic", required=False, action="store", type=int, default=15)
    parser.add_argument("--spr", required=False, action="store", type=int, default=5)
    args = parser.parse_args()
    output_folder = args.output_folder

    versions = args.versions
    if versions > 26:
        versions = 26
    ic_sample_size = args.ic
    spr_sample_size = args.spr

    versies = [x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:versions]]

    gdf = gpd.read_file(INPUT_FILE)
    lines_gdf = gdf[gdf.geometry.geom_type.isin({"LineString", "MultiLineString"})]
    stations_gdf = gpd.read_file(INPUT_OV_FILE)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    MAIN_TYPES = {"rail"}
    lines_main = lines_gdf[lines_gdf["railway"].isin(MAIN_TYPES)].copy()

    # Stations / halts only
    STATION_TYPES = {"Treinstation"}
    stations = stations_gdf[stations_gdf["Type_halte"].isin(STATION_TYPES)].copy()
    stations = stations.drop_duplicates(subset="Naam", keep="first")
    stations = stations.dropna(subset=["geometry"])

    # clean those names up
    overhoor_stations = stations.copy()
    overhoor_stations["Naam"] = overhoor_stations["Naam"].apply(lambda x: split_name(x))
    stations["Naam"] = stations["Naam"].apply(lambda x: split_name(x))

    provinces = load_admin_borders()

    for versie in versies:
        output_file = os.path.join(output_folder, f"{versie}_overhoring.jpg")
        output_file_legend = os.path.join(output_folder, f"{versie}_antwoorden.jpg")
        make_files(
            versie,
            output_file,
            output_file_legend,
            lines_main,
            stations,
            overhoor_stations,
            provinces,
            ic_sample_size,
            spr_sample_size,
        )


if __name__ == "__main__":
    main()
