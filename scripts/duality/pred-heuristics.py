import argparse
import os
import pathlib
import shutil
import tempfile

import numpy as np
from midynet.config import Config, ExperimentConfig
from midynet.scripts import ScriptManager

from util import format_sequence


class Figure3PredHeuristicsConfig(ExperimentConfig):
    @classmethod
    def default(
        cls,
        path_to_data=None,
        n_workers=1,
        n_async_jobs=1,
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
            f"predheur-glauber-erdosrenyi",
            "glauber",
            "erdosrenyi",
            metrics=[
                "pred_error",
            ],
            path=path_to_data,
            n_workers=n_workers,
            n_async_jobs=n_async_jobs,
            seed=seed,
        )
        config.path += "/" + config.name
        config.data_model.coupling = format_sequence(
            (0, 0.2, 20), (0.2, 0.8, 10)
        )

        config.data_model.length = 100
        config.prior.size = 100
        config.prior.edge_count = 250
        config.prior.loopy = config.prior.multigraph = False
        config.metrics.pred_error.predictor = [
            "average_probability",
            # "mle",
            "logistic",
            "mlp",
        ]
        config.metrics.pred_error.n_samples = n_workers // n_async_jobs
        config.metrics.pred_error.measures = "absolute_error"
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

    if not os.path.exists("./configs"):
        os.mkdir("./configs")
    if not os.path.exists("./log"):
        os.mkdir("./log")
    config = Figure3PredHeuristicsConfig.default(
        path_to_data="./data",
        n_workers=64,
        n_async_jobs=4,
        seed=None,
        time="20:00:00",
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
        # "run": "Recon heuristics vs recon on erdosrenyi large",
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
