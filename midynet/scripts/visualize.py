import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pyhectiqlab

from typing import List, Optional, Tuple

import midynet
from midynet.utility import display
from midynet.statistics import Statistics

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


def get_stat(
    path: str,
    y_axis: str,
    x_axis: str,
    aux_axis: Optional[str] = None,
    name: Optional[str] = None,
    metrics: Optional[str] = None,
) -> Tuple[Statistics, pd.Series, pd.Series or None]:
    df = None
    if metrics is not None:
        path_to_metrics = os.path.join(path, m.shortname + ".pkl")
    else:
        for m in midynet.metrics.__all_metrics__:
            path_to_metrics = os.path.join(path, m.shortname + ".pkl")
            if y_axis in m.keys and os.path.exists(path_to_metrics):
                assert (
                    df is None
                ), f"{metrics} has a duplicate ({m}) and thus cannot be infered: please, specify metrics name."
                metrics = m
                df = pd.read_pickle(path_to_metrics)
    assert df is not None
    if isinstance(df, dict) and name is not None:
        df = df[name]
    else:
        assert len(df) == 1
        df = next(iter(df.values()))
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
    subconfig: str = None,
    run: Optional[str] = None,
    path_to_figure=None,
    metrics: Optional[str] = None,
    **kwargs,
):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for i, _y in enumerate(y_axis):
        y, x, aux = get_stat(
            path,
            _y,
            x_axis,
            aux_axis=aux_axis,
            name=subconfig,
            metrics=metrics,
        )

        if aux is not None:
            palette = sb.color_palette(list(display.dark_colors.values()), as_cmap=True)
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
            legend="full",
            markers=False,
        )
    if len(y_axis) > 1:
        ax.set_ylabel("/".join(y_axis), fontsize=FONTSIZE)
    else:
        ax.set_ylabel(y_axis[0], fontsize=FONTSIZE)
    ax.set_xlabel(x.name, fontsize=FONTSIZE)
    if twinx_axis is not None:
        _ax = ax.twinx()
        for i, _y in enumerate(twinx_axis):
            y, x, aux = get_stat(path, _y, x_axis, aux_axis=aux_axis, name=subconfig)
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
                markers=False,
            )

        if len(twinx_axis) > 1:
            _ax.set_ylabel("/".join(twinx_axis), fontsize=FONTSIZE)
        else:
            _ax.set_ylabel(twinx_axis[0], fontsize=FONTSIZE)
    fig.tight_layout()
    print(path_to_figure)
    if path_to_figure is not None:
        fig.savefig(path_to_figure)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        "-r",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--path",
        "-p",
        help="Path to experiment.",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--version",
        "-v",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--subconfig",
        type=str,
        default=None,
        required=False,
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
        "--figure",
        "-f",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="png",
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
    assert args.path is not None or args.name is not None
    if args.path is None:
        args.path = pyhectiqlab.datasets.download_dataset(
            args.name,
            project_path="dynamica/midynet",
            version=args.version,
            save_path=args.save_path,
        )
    if args.figure is not None:
        path_to_figure = os.path.join(
            args.save_path, args.figure + "." + args.extension
        )
    else:
        path_to_figure = None
    main(
        path_to_figure=path_to_figure,
        **args.__dict__,
    )
