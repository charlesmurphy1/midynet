import os
import argparse
import pyhectiqlab
import logging
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

import midynet
import matplotlib.pyplot as plt


def get_stats(df, name: None, separator="-"):
    keys = []
    d = {}
    for c in df.columns:
        n, t = c.split(separator)

        if n == name or name is None:
            d[t] = df[c].values
    return midynet.Statistics(d)


def format_data(metrics, params, names=[], xaxis=None):
    x = (
        np.unique(params.get(xaxis).values)
        if xaxis is not None
        else np.array(params.index)
    )

    y = {}
    for n in names:
        for _x in x:
            yy = defaultdict(lambda: defaultdict(lambda: list))
            for i in np.where(params.get(xaxis).values == _x)[0]:
                print(i)
                key = ", ".join(
                    [
                        f"{k}={v}"
                        for k, v in dict(params.loc[i]).items()
                        if k != xaxis
                    ]
                )
                yy[key]["mid"].append(metrics[n][i]["mid"])
                yy[key]["low"].append(metrics[n][i]["low"])
                yy[key]["high"].append(metrics[n][i]["high"])
        y[n] = {k: midynet.Statistics(v) for k, v in yy.items()}
    print(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the information measures."
    )

    parser.add_argument(
        "--name",
        "-n",
        metavar="RUN_NAME",
        help="Name of the run.",
        default=None,
    )
    parser.add_argument(
        "--path_to_metrics",
        "-p",
        type=str,
        metavar="PATH_TO_METRICS",
        help="Path to the metrics file (.pkl).",
        default=None,
    )
    parser.add_argument(
        "--artifact_uuid",
        "-i",
        type=str,
        metavar="ARTIFACT_UUID",
        help="Artifact ID from the lab.",
        default=None,
    )
    parser.add_argument(
        "--xaxis",
        "-x",
        help="X-axis to use.",
        default=None,
    )
    parser.add_argument(
        "--show",
        "-s",
        help="Shows the generated figures.",
        action="store_true",
    )

    args = parser.parse_args()
    if args.path_to_metrics is None and args.artifact_uuid is None:
        msg = f"'path_to_metrics' and 'artifact_uuid' cannot be both 'None'."
    elif args.artifact_uuid is not None:
        from pyhectiqlab.utils import download_existing_run_artifact
        import tempfile

        tmp = tempfile.mktemp()
        os.mkdir(tmp)
        args.path_to_metrics = download_existing_run_artifact(
            artifact_uuid=args.artifact_uuid, savepath=tmp
        )

    data = pickle.load(open(args.path_to_metrics, "rb"))
    for config_name, _data in data.items():
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        metrics_names = set(
            [name.split("-")[0] for name in _data["metrics"].columns]
        )
        metrics = {
            name: get_stats(_data["metrics"], name=name)
            for name in metrics_names
        }
        params = pd.DataFrame(_data["params"])
        # xaxis = (
        #     np.unique(params.get(args.x_axis).values)
        #     if args.x_axis is not None
        #     else np.array(params.index)
        # )
        # print(xaxis)
        # midynet.utility.display.plot_statistics()
        print(
            format_data(
                metrics,
                params,
                names=metrics_names,
                xaxis=args.xaxis,
            )
        )

    # print(f"{metrics=}")
    # print(f"{params=}")
