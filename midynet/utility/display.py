import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from typing import Optional, Union
from cycler import cycler
from palettable.palette import Palette

from .convert import get_edgelist, convert_basegraph_to_networkx

__all__ = (
    "fontsizes",
    "hex_to_rgb",
    "rgb_to_hex",
    "palettes",
    "markers",
    "linestyles",
    "cycle",
    "Label",
)

fontsizes = {"small": 6, "medium": 8, "large": 10}


def hex_to_rgb(value):
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


def setup_dir(path: pathlib.Path = "."):
    path = pathlib.Path(path) if isinstance(path, str) else path
    path_to_svg = path / "svg"
    path_to_svg.mkdir(exist_ok=True, parents=True)
    path_to_pdf = path / "pdf"
    path_to_pdf.mkdir(exist_ok=True, parents=True)
    path_to_png = path / "png"
    path_to_png.mkdir(exist_ok=True, parents=True)


def clean_dir(path: pathlib.Path = ".", prefix: str = None):
    path = pathlib.Path(path) if isinstance(path, str) else path
    for path_to in [path / "svg", path / "pdf", path / "png"]:
        for local, subpaths, files in os.walk(path_to):
            for f in files:
                print(local, f)
                name = f.split(".")[0]
                if prefix is None or prefix == name[: len(prefix)]:
                    p = pathlib.Path(local) / f
                    p.unlink()


def drawHierarchyTree(b):
    hierarchy = nx.Graph()
    b = list(reversed(b))
    for l in range(1, len(b)):
        for i, bb in enumerate(b[l]):
            hierarchy.add_edge(f"l{l-1}-{bb}", f"l{l}-{i}")
    pos = nx.drawing.nx_pydot.graphviz_layout(hierarchy, prog="dot")
    nx.draw(hierarchy, pos, with_labels=True)
    plt.show()


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
all_color_sequence = (
    [c for c in light_colors.values()]
    + [c for c in med_colors.values()]
    + [c for c in dark_colors.values()]
)

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
    palettes[k] = Palette(k, "diverging", cm)
    palettes[k] = Palette("inv_" + k, "diverging", cm[::-1])

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

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=fontsizes["medium"])
plt.rc("lines", linewidth=1)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rc("axes", prop_cycle=cycle)


class Label:
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"
    locations = {
        "center center": (0.5, 0.5, "center", "center"),
        "top right": (0.95, 0.95, "top", "right"),
        "bottom right": (0.95, 0.05, "bottom", "right"),
        "top left": (0.05, 0.95, "top", "left"),
        "bottom left": (0.05, 0.05, "bottom", "left"),
    }
    counter: int = 0
    type: str = "bold"
    left: str = "("
    right: str = ")"

    def __init__(self):
        raise NotImplementedError()

    @classmethod
    def init(cls):
        cls.counter = 0

    @classmethod
    def clear(cls):
        cls.counter = 0

    @classmethod
    def getLabel(cls):
        label = cls.alphabet[cls.counter]
        cls.counter += 1
        if cls.type == "normal":
            return rf"{cls.left}{label}{cls.right}"
        elif cls.type == "bold":
            return rf"\textbf{{ {cls.left}{label}{cls.right} }}"
        elif cls.type == "italic":
            return rf"\textit{{ {cls.left}{label}{cls.right} }}"

    @classmethod
    def plot(
        cls,
        ax,
        label: Optional[str] = None,
        loc: Union[tuple, str] = "center center",
        fontsize: int = fontsizes["large"],
        box: bool = True,
    ):
        label = cls.getLabel() if label is None else label
        if isinstance(loc, tuple):
            h, v, va, ha = loc
        elif isinstance(loc, str):
            h, v, va, ha = cls.locations[loc]

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
        return ax


def drawPieMarker(xs, ys, ratios, colors, size=60, ax=None):
    assert sum(ratios) <= 1, "sum of ratios needs to be < 1"

    markers = []
    previous = 0
    # calculate the points of the pie pieces
    for color, ratio in zip(colors, ratios):
        this = 2 * np.pi * ratio + previous
        x = [0] + np.cos(np.linspace(previous, this, 10)).tolist() + [0]
        y = [0] + np.sin(np.linspace(previous, this, 10)).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        markers.append(
            {
                "marker": xy,
                "s": np.abs(xy).max() ** 2 * np.array(size),
                "facecolor": color,
            }
        )

    # scatter each of the pie pieces to create pies
    for marker in markers:
        ax.scatter(xs, ys, zorder=10, **marker)


def draw_graph(
    graph,
    labels=None,
    ax=None,
    pos=None,
    marginals=None,
    ec=med_colors["grey"],
    ew=1,
    with_self_loops=True,
    with_parallel_edges=True,
):
    ax = plt.gca() if ax is None else ax
    labels = [0] * graph.get_size() if labels is None else labels
    el = []
    colors = (
        [c for c in light_colors.values()]
        + [c for c in med_colors.values()]
        + [c for c in dark_colors.values()]
    )
    nx_graph = convert_basegraph_to_networkx(graph)
    if not with_self_loops:
        nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    if pos is None:
        pos = nx.spring_layout(nx_graph)

    nx.draw_networkx_edges(
        nx_graph,
        pos=pos,
        edgelist=get_edgelist(graph),
        width=[
            ew * graph.get_edge_multiplicity_idx(*e) if with_parallel_edges else ew
            for e in get_edgelist(graph)
        ],
        edge_color=ec,
        ax=ax,
    )

    for i, p in pos.items():
        if marginals is None:
            ax.plot([p[0]], [p[1]], color=colors[labels[i]], marker="o")
        else:
            drawPieMarker([p[0]], [p[1]], marginals[i], colors=colors, ax=ax)


def plot_statistics(
    x,
    y,
    ax=None,
    fill=True,
    fill_alpha=0.2,
    fill_color=None,
    bar=True,
    spacing=1,
    interpolate=None,
    interp_num_points=1000,
    error_scaling=1,
    **kwargs,
):
    ax = plt.gca() if ax is None else ax
    c = kwargs.get("color", "grey")
    a = kwargs.get("alpha", 1)
    index = np.argsort(x)
    x = np.array(x)
    marker = kwargs.pop("marker", markers[0])
    linestyle = kwargs.pop("linestyle", linestyles[0])
    kwargs.pop("ls", None)
    label = kwargs.pop("label", None)

    if interpolate is not None:
        interpF = interp1d(x, y["mid"], kind=interpolate)
        interpX = np.linspace(min(x), max(x), interp_num_points)
        interpY = interpF(interpX)
        interpErrorLowF = interp1d(
            x,
            y["mid"] - np.abs(y["low"]) / error_scaling,
            kind=interpolate,
        )
        interpErrorHighF = interp1d(
            x,
            y["mid"] + np.abs(y["high"]) / error_scaling,
            kind=interpolate,
        )
        interpErrorLowY = interpErrorLowF(interpX)
        interpErrorHighY = interpErrorHighF(interpX)

        ax.plot(interpX, interpY, marker="None", linestyle=linestyle, **kwargs)

        if fill:
            fill_color = c if fill_color is None else fill_color
            ax.fill_between(
                interpX,
                interpErrorLowY,
                interpErrorHighY,
                color=fill_color,
                alpha=a * fill_alpha,
                linestyle="None",
            )
    else:
        ax.plot(
            x[index],
            y["mid"][index],
            marker="None",
            linestyle=linestyle,
            **kwargs,
        )
        if fill:
            fill_color = c if fill_color is None else fill_color
            ax.fill_between(
                x[index],
                y["mid"][index] - np.abs(y["low"][index]) / error_scaling,
                y["mid"][index] + np.abs(y["high"][index]) / error_scaling,
                color=fill_color,
                alpha=a * fill_alpha,
                linestyle="None",
            )
    ax.plot(
        x[index[::spacing]],
        y["mid"][index[::spacing]],
        marker=marker,
        linestyle="None",
        **kwargs,
    )

    if label is not None:
        ax.plot(
            x[0], y["mid"][0], marker=marker, linestyle=linestyle, label=label, **kwargs
        )

    if bar:
        ax.errorbar(
            x[index],
            y["mid"][index],
            np.vstack((np.abs(y["low"][index]), np.abs(y["high"][index])))
            / error_scaling,
            ecolor=c,
            marker="None",
            linestyle="None",
            **kwargs,
        )
    return ax


def main():
    for k, p in palettes.items():
        p.save_discrete_image("discrete-palettes/" + p.name + ".png")
        p.save_continuous_image("continuous-palettes/" + p.name + ".png")


if __name__ == "__main__":
    main()
