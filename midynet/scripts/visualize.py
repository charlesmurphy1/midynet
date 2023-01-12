import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from typing import List, Optional, Tuple

import midynet
from midynet.utility import display
from midynet.statistics import Statistics

xaxis_lexicon = {
    "infection_prob": r"Infection prob.",
    "edge_count": r"Edge count",
}
yaxis_lexicon = {
    "likelihood": r"$H(X|G)$",
    "prior": r"$H(G)$",
    "posterior": r"$H(G|X)$",
    "evidence": r"$H(X)$",
    "mutualinfo": r"$I(X;G)$",
    "recon": r"$U(G|X)$",
    "pred": r"$U(X|G)$",
    "auc": r"AUC",
}

FONTSIZE = 16


def infer_complete_name(name: str, df: Optional[pd.DataFrame] = None):
    if df is not None and name in df:
        return name
    if df is None:
        return name
    for c in df.columns:
        if name == c.split(".")[-1]:
            return c
    return None


# def stat_lineplot(
#     stat: Statistics,
#     x_series: pd.Series,
#     aux_series: Optional[pd.Series] = None,
#     **kwargs
# ):
#     bs = stat.bootstrap(kwargs.pop("num_samples", 1000))
#     df = pd.DataFrame.from_records(bs)
#     df[x_series.name] = x_series
#     if aux_series is not None:
#         df[aux_series.name] = aux_series

#     df = df.melt(
#         [x_series.name, aux_series.name]
#         if aux_series is not None
#         else x_series.name
#     )
#     return sb.lineplot(
#         df,
#         y="value",
#         x=x_series.name,
#         hue=aux_series.name if aux_series is not None else None,
#         **kwargs
#     )


def get_stat(
    path: str,
    y_axis: str,
    x_axis: str,
    aux_axis: Optional[str] = None,
    name: Optional[str] = None,
) -> Tuple[Statistics, pd.Series, pd.Series or None]:
    df = None
    for m in midynet.metrics.__all_metrics__:
        if y_axis in m.keys:
            df = pd.read_pickle(os.path.join(path, m.shortname + ".pkl"))
    assert df is not None
    if isinstance(df, dict) and name is not None:
        df = df[name]
    x = df[infer_complete_name(x_axis, df)]
    aux_axis = infer_complete_name(aux_axis, df)
    aux = df[aux_axis] if aux_axis is not None else None
    stat = Statistics.from_dataframe(df, key=y_axis)
    return stat, x, aux


def main(
    path: str,
    y_axis: str or List[str],
    x_axis: str,
    twinx_axis: str or List[str],
    aux_axis: Optional[str] = None,
    save_path: str = "./",
    name: str = None,
):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for i, _y in enumerate(y_axis):
        y, x, aux = get_stat(path, _y, x_axis, aux_axis=aux_axis, name=name)

        if aux is not None:
            palette = sb.color_palette(
                list(display.dark_colors.values()), as_cmap=True
            )
            color = None
        else:
            color = list(display.dark_colors.values())[i]
            palette = None
        y.lineplot(
            x=x,
            aux=aux,
            ax=ax,
            color=color,
            palette=palette,
            # markers=display.markers,
            # label=_y if len(y_axis) > 1 else None,
            legend="full",
            markers=False,
        )
        # _ax.set_ylabel(yaxis_lexicon[_y])
    if len(y_axis) > 1:
        # ax.legend(loc=0, fontsize=FONTSIZE)
        ax.set_ylabel("/".join(y_axis), fontsize=FONTSIZE)
    else:
        ax.set_ylabel(y_axis[0], fontsize=FONTSIZE)
    ax.set_xlabel(x.name, fontsize=FONTSIZE)
    if twinx_axis is not None:
        _ax = ax.twinx()
        for i, _y in enumerate(twinx_axis):
            y, x, aux = get_stat(
                path, _y, x_axis, aux_axis=aux_axis, name=name
            )
            if aux is not None:
                palette = sb.color_palette(
                    list(display.light_colors.values()), as_cmap=True
                )
                color = None
            else:
                color = list(display.light_colors.values())[i]
                palette = None
            y.lineplot(
                x=x,
                aux=aux,
                ax=_ax,
                color=color,
                palette=palette,
                # markers=display.markers,
                # label=_y if len(twinx_axis) > 1 else None,
                markers=False,
                legend="full",
            )

        if len(twinx_axis) > 1:
            # _ax.legend(loc=0, fontsize=FONTSIZE)
            _ax.set_ylabel("/".join(twinx_axis), fontsize=FONTSIZE)
        else:
            _ax.set_ylabel(twinx_axis[0], fontsize=FONTSIZE)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        "-p",
        help="Path to experiment.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        help="Name of the config.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        help="Path where the figure is saved.",
        default="./",
        required=False,
    )
    parser.add_argument(
        "--y_axis",
        "-y",
        type=str,
        help="Name of the yaxis",
        nargs="*",
        required=True,
    )
    parser.add_argument(
        "--x_axis",
        "-x",
        type=str,
        help="Name of the x-axis.",
        required=True,
    )
    parser.add_argument(
        "--twinx_axis",
        "-t",
        type=str,
        default=None,
        required=False,
        nargs="*",
    )
    parser.add_argument(
        "--aux_axis",
        "-a",
        type=str,
        default=None,
        help="Name of any other auxilary axis to include in the legend.",
        required=False,
    )

    args = parser.parse_args()
    main(**args.__dict__)
