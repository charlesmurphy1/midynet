import os
import argparse
import pandas as pd

import midynet

old_to_new = dict(
    mi="mutualinfo",
    hxg="likelihood",
    hx="evidence",
    hg="prior",
    hgx="posterior",
)
new_to_old = {n: o for o, n in old_to_new.items()}


def convert(old_data, key):
    new_data = {}
    new_data[key + "_loc"] = old_data[new_to_old[key] + "-mid"]
    new_data[key + "_scale"] = 0.5 * (
        old_data[new_to_old[key] + "-low"] + old_data[new_to_old[key] + "-high"]
    )
    return new_data


def main(path_to_data: str):
    old_data = pd.read_pickle(os.path.join(path_to_data, "mutualinfo.pickle"))

    new_data = {}
    for k, v in old_data.items():
        new_data[k] = {}
        for o, n in old_to_new.items():
            new_data[k].update(convert(v, n))
        new_data[k] = pd.DataFrame(new_data[k])
        mi = midynet.statistics.Statistics.from_dataframe(new_data[k], "mutualinfo")
        prior = midynet.statistics.Statistics.from_dataframe(new_data[k], "prior")
        evidence = midynet.statistics.Statistics.from_dataframe(new_data[k], "evidence")
        recon = mi / prior
        pred = mi / evidence
        new_data[k]["recon_loc"] = recon.loc
        new_data[k]["recon_scale"] = recon.scale
        new_data[k]["pred_loc"] = pred.loc
        new_data[k]["pred_scale"] = pred.scale
    pd.to_pickle(new_data, os.path.join(path_to_data, "reconinfo.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_data",
        "-p",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(**args.__dict__)
