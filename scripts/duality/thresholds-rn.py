import argparse
import os
import pathlib
import shutil
import tempfile

import numpy as np
from midynet.config import DataModelConfig, ExperimentConfig, GraphConfig
from midynet.scripts import ScriptManager


def format_sequence(*arr):
    arr = [
        np.linspace(*s) if isinstance(s, tuple) and len(s) == 3 else s
        for s in arr
    ]
    return np.unique(np.concatenate(arr)).tolist()


couplings = {
    "glauber": format_sequence((0, 0.02, 20), (0.02, 0.08, 10)),
    "sis": format_sequence((0, 0.02, 20), (0.02, 0.2, 10), (0.2, 1.0, 10)),
    "cowan_forward": format_sequence(
        (0, 0.07, 5), (0.07, 0.2, 30), (0.2, 0.3, 5)
    ),
    "cowan_backward": format_sequence((0, 0.1, 25), (0.1, 0.3, 15)),
}
STEP_FACTOR = 4

graph_dict = {
    # "glauber": ("littlerock", "/home/murphy9/data/graphs/littlerock.npy"),
    "glauber": ("polblogs", "/home/murphy9/data/graphs/polblogs.npy"),
    "sis": ("euairlines", "/home/murphy9/data/graphs/euairlines.npy"),
    "cowan_forward": ("celegans", "/home/murphy9/data/graphs/celegans.npy"),
    "cowan_backward": ("celegans", "/home/murphy9/data/graphs/celegans.npy"),
}

model_dict = {
    "glauber": DataModelConfig.glauber(
        length=2000, coupling=couplings["glauber"]
    ),
    "sis": DataModelConfig.sis(
        length=2000, infection_prob=couplings["sis"], recovery_prob=0.5
    ),
    "cowan_forward": DataModelConfig.cowan_forward(
        length=2000, nu=couplings["cowan_forward"]
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
        num_workers=1,
        time="24:00:00",
        mem=12,
        seed=None,
    ):

        path_to_data = pathlib.Path(
            tempfile.mktemp() if path_to_data is None else path_to_data
        )
        if not os.path.exists(path_to_data):
            os.makedirs(path_to_data)
        config = ExperimentConfig.default(
            f"thresholds-{graph_dict}-{model}",
            data_model,
            "nbinom",
            metrics=[
                "susceptibility",
            ],
            path=path_to_data,
            num_workers=num_workers,
            seed=seed,
        )
        N = 1000
        M = 2500
        T = 10000
        if data_model == "sis":
            config.data_model.infection_prob = np.unique(
                np.concatenate(
                    [
                        np.linspace(0.0, 0.1, 30),
                        np.linspace(0.1, 0.5, 10),
                    ]
                )
            ).tolist()
            config.data_model.recovery_prob = 0.5
        elif data_model == "glauber":
            config.data_model.coupling = np.unique(
                np.concatenate(
                    [
                        np.linspace(0.0, 0.05, 5),
                        np.linspace(0.05, 0.15, 20),
                        np.linspace(0.15, 0.5, 10),
                    ]
                )
            ).tolist()
        elif data_model == "cowan_backward":
            config.data_model.nu = np.unique(
                np.concatenate(
                    [
                        np.linspace(0.1, 0.2, 5),
                        np.linspace(0.2, 0.4, 30),
                        np.linspace(0.4, 0.8, 5),
                    ]
                )
            ).tolist()
            config.data_model.eta = 0.5
        elif data_model == "cowan_forward" or data_model == "cowan":
            config.data_model.nu = np.unique(
                np.concatenate(
                    [
                        np.linspace(0.1, 0.3, 5),
                        np.linspace(0.3, 0.5, 30),
                        np.linspace(0.5, 0.8, 5),
                    ]
                )
            ).tolist()
            config.data_model.eta = 0.5
        config.data_model.length = T

        config.prior.size = N
        config.prior.edge_count = M
        config.prior.heterogeneity = 1
        config.metrics.susceptibility.num_samples = 1 * num_workers
        config.resources.update(
            account="def-aallard",
            time=time,
            mem=f"{mem}G",
            cpus_per_task=config.num_workers,
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
    for data_model in ["cowan_forward", "cowan_backward"]:
        config = ThresholdConfig.default(
            data_model,
            num_workers=12,
            time="24:00:00",
            mem=12,
            path_to_data=f"./data/threshold-{data_model}",
        )
        if args.overwrite and os.path.exists(config.path):
            shutil.rmtree(config.path)
            os.makedirs(config.path)
        path_to_config = f"./configs/{config.name}.pkl"
        config.save(path_to_config)
        script = ScriptManager(
            executable="python ../../midynet/scripts/recon.py",
            execution_command="bash",
            path_to_scripts="./scripts",
        )
        extra_args = {
            # "run": "Pred heuristics vs pred on erdosrenyi sis large",
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
                "python/3.8",
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
