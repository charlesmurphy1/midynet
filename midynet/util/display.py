import h5py

import matplotlib.pyplot as plt
import numpy as np
import os

from palettable.palette import Palette
from cycler import cycler


def hex_to_rgb(value):
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


locations = {
    "center center": (0.5, 0.5, "center", "center"),
    "upper right": (0.95, 0.95, "top", "right"),
    "lower right": (0.95, 0.05, "bottom", "right"),
    "upper left": (0.05, 0.95, "top", "left"),
    "lower left": (0.05, 0.05, "bottom", "left"),
}

hex_colors = {
    "blue": ["#7bafd3", "#1f77b4", "#1f53ff"],
    "orange": ["#f7be90", "#f19143", "#fd7931"],
    "red": ["#e78580", "#d73027", "#b4190c"],
    "purple": ["#c3b4d6", "#9A80B9", "#9a3fb9"],
    "green": ["#9fdaac", "#33b050", "#14960f"],
    "grey": ["#999999", "#525252", "#323232"],
    "black": ["#000000"] * 3,
    "white": ["#ffffff"] * 3,
}

rgb_colors = {k: list(map(hex_to_rgb, v)) for k, v in hex_colors.items()}
light_colors = {k: v[0] for k, v in hex_colors.items()}
med_colors = {k: v[1] for k, v in hex_colors.items()}
dark_colors = {k: v[2] for k, v in hex_colors.items()}

palettes = {}

for k in hex_colors.keys():
    cm = [rgb_colors["white"][0], *rgb_colors[k], rgb_colors["black"][0]]
    if k != "black" and k != "white":
        palettes[k + "s"] = Palette(k + "s", "sequential", cm)
        palettes["inv_" + k + "s"] = Palette("inv_" + k + "s", "sequential", cm[::-1])

for i, k in enumerate(["light", "med", "dark"]):
    cm = [
        rgb_colors["orange"][i],
        rgb_colors["red"][i],
        rgb_colors["purple"][i],
        rgb_colors["blue"][i],
        rgb_colors["green"][i],
    ]
    # palettes[k] = Palette(k, "diverging", cm)
    # palettes[k] = Palette("inv_" + k, "diverging", cm[::-1])

markers = ["o", "s", "v", "^", "*", "d"]
linestyles = [
    "solid",
    "dashed",
    "dotted",
    "dashdot",
    (0, (5, 5)),
    (0, (3, 5, 1, 5, 1, 5)),
]

cycle = cycler(
    color=list(dark_colors.values())[:-2], marker=markers, linestyle=linestyles
)
large_fontsize = 18
small_fontsize = 14


plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=small_fontsize)
plt.rc("lines", linewidth=2)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rc("axes", prop_cycle=cycle)


def label_plot(ax, label, loc="center center", fontsize=18, box=None):
    if isinstance(loc, tuple):
        h, v, va, ha = loc
    elif isinstance(loc, str):
        h, v, va, ha = locations[loc]
    if isinstance(box, bool):
        box = dict(boxstyle="round", color="white", alpha=0.75) if box else None

    ax.text(
        h,
        v,
        label,
        color="k",
        transform=ax.transAxes,
        verticalalignment=va,
        horizontalalignment=ha,
        fontsize=fontsize,
        bbox=box,
    )


def main():
    for k, p in palettes.items():
        p.save_discrete_image("discrete-palettes/" + p.name + ".png")
        p.save_continuous_image("continuous-palettes/" + p.name + ".png")


if __name__ == "__main__":
    main()
