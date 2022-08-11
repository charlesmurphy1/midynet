import os
import pathlib
import matplotlib.pyplot as plt
import networkx as nx

from typing import Optional, Union
from cycler import cycler
from palettable.palette import Palette

from .convert import get_edgelist, get_networkx_graph_from_basegraph

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


def draw_graph(graph, labels=None, ax=None, pos=None, marginals=None):
    ax = plt.gca() if ax is None else ax
    labels = [0] * graph.get_size() if labels is None else labels
    el = []
    colors = (
        [c for c in light_colors.values()]
        + [c for c in med_colors.values()]
        + [c for c in dark_colors.values()]
    )
    nx_graph = get_networkx_graph_from_basegraph(graph)
    if pos is None:
        pos = nx.spring_layout(nx_graph)

    # for i, j in get_edgelist(graph):
    #     pos_i, pos_j = pos[i], pos[j]
    #     ax.plot(
    #         [pos_i[0], pos_j[0]],
    #         [pos_i[1], pos_j[1]],
    #         linestyle="-",
    #         marker="None",
    #         color=med_colors["grey"],
    #         linewidth=graph.get_edge_multiplicity_idx(i, j),
    #     )
    nx.draw_networkx_edges(
        nx_graph,
        pos=pos,
        edgelist=get_edgelist(graph),
        width=[graph.get_edge_multiplicity_idx(*e) for e in get_edgelist(graph)],
    )

    for i, p in pos.items():
        if marginals is None:
            ax.plot([p[0]], [p[1]], color=colors[labels[i]], marker="o")
        else:
            drawPieMarker([p[0]], [p[1]], marginals[i], colors=colors, ax=ax)


def main():
    for k, p in palettes.items():
        p.save_discrete_image("discrete-palettes/" + p.name + ".png")
        p.save_continuous_image("continuous-palettes/" + p.name + ".png")


if __name__ == "__main__":
    main()
