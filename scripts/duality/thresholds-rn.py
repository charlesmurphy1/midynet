import argparse
import os
import pathlib
import shutil
import tempfile

import numpy as np
from math import ceil
from midynet.config import DataModelConfig, ExperimentConfig, GraphConfig
from midynet.scripts import ScriptManager

PATH_TO_GRAPHS = "../../data/graphs/"
PATH_TO_DATA = "./data"
COMMAND = "bash"


def format_sequence(*arr):
    arr = [
        np.linspace(*s) if isinstance(s, tuple) and len(s) == 3 else s
        for s in arr
    ]
    return np.unique(np.concatenate(arr)).tolist()


couplings = {
    "glauber": format_sequence((0, 0.02, 20), (0.02, 0.08, 10)),
    "sis": format_sequence((0, 0.02, 20), (0.02, 0.1, 10)),
    "cowan_backward": format_sequence((0, 0.1, 25), (0.1, 0.3, 15)),
    "cowan_forward": format_sequence((0, 0.1, 25), (0.1, 0.3, 15)),
}
# PATH_TO_GRAPHS = "/home/murphy9/data/graphs/"

graph_dict = {
    "glauber": ("littlerock", f"{PATH_TO_GRAPHS}/littlerock.npy"),
    # "glauber": ("polblogs", f"{PATH_TO_GRAPHS}/polblogs.npy"),
    "sis": ("euairlines", f"{PATH_TO_GRAPHS}/euairlines.npy"),
    "cowan_forward": ("celegans", f"{PATH_TO_GRAPHS}/celegans.npy"),
    "cowan_backward": ("celegans", f"{PATH_TO_GRAPHS}/celegans.npy"),
}

model_dict = {
    "glauber": DataModelConfig.glauber(
        length=2000, coupling=couplings["glauber"]
    ),
    "sis": DataModelConfig.sis(
        length=2000, infection_prob=couplings["sis"], recovery_prob=0.8
    ),
    "cowan_forward": DataModelConfig.cowan_forward(
        length=10000, nu=couplings["cowan_forward"]
    ),
    "cowan_backward": DataModelConfig.cowan_backward(
        length=2000, nu=couplings["cowan_backward"]
    ),
}


class ThresholdWithRealNetworksConfig(ExperimentConfig):
    @classmethod
    def default(
        cls,
        model,
        path_to_data=None,
        n_workers=1,
        time="24:00:00",
        mem=12,
        seed=None,
    ):
        path_to_data = pathlib.Path(
            tempfile.mktemp() if path_to_data is None else path_to_data
        )
        if not os.path.exists(path_to_data):
            os.makedirs(path_to_data)
        target, target_path = graph_dict[model]
        assert os.path.exists(
            target_path
        ), f"path {target_path} does not exist."
        assert os.path.exists(
            path_to_data
        ), f"path {path_to_data} does not exist."
        config = ExperimentConfig.default(
            f"thresholds-{target}-{model}",
            model,
            "degree_constrained_configuration",
            target=target,
            metrics=[
                "susceptibility",
            ],
            path=path_to_data,
            n_workers=n_workers,
            seed=seed,
        )
        config.data_model = model_dict[model]

        config.prior.size = config.target.size
        if "backward" in model:
            config.data_model.n_active = config.prior.size
        if model == "glauber":
            config.data_model.n_active = -1
        else:
            config.data_model.n_active = ceil(0.01 * config.prior.size)
        config.metrics.susceptibility.n_samples = 1 * n_workers
        config.metrics.susceptibility.resample_graph = True
        config.resources.update(
            account="def-aallard",
            time=time,
            mem=f"{mem}G",
            cpus_per_task=config.n_workers,
            job_name=config.name,
            output=f"log/{config.name}.out",
        )
        config.lock()
        return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
    )
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
    )
    args = parser.parse_args()
    for data_model in ["glauber"]:
        config = ThresholdWithRealNetworksConfig.default(
            data_model,
            n_workers=8,
            time="24:00:00",
            mem=12,
            path_to_data=f"{PATH_TO_DATA}/thresholds/threshold-rn-{data_model}",
        )
        if args.overwrite and os.path.exists(config.path):
            shutil.rmtree(config.path)
            os.makedirs(config.path)
        path_to_config = f"./configs/{config.name}.pkl"
        config.save(path_to_config)
        script = ScriptManager(
            executable="python ../../midynet/scripts/recon.py",
            execution_command=COMMAND,
            path_to_scripts="./scripts",
        )
        extra_args = {
            "name": config.name,
            "path_to_config": path_to_config,
            "resume": args.resume,
            "save_patience": 1,
        }
        script.run(
            name=config.name,
            modules_to_load=[
                "StdEnv/2020",
                "gcc/9",
                "python/3.9",
                "graph-tool",
                "scipy-stack",
                "httpproxy",
            ],
            virtualenv="/home/murphy9/.midynet-env/bin/activate",
            extra_args=extra_args,
            resources=config.resources.dict,
        )


if __name__ == "__main__":
    main()
