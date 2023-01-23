import numpy as np
import os
import tempfile
import pathlib
import argparse
import shutil

from midynet.config import Config, ExperimentConfig
from midynet.scripts import ScriptManager


class Figure2HeuristicsConfig(ExperimentConfig):
    @classmethod
    def default(
        cls,
        prior="erdosrenyi",
        data_model="glauber",
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
        config = cls.reconstruction(
            f"heuristics-{prior}-{data_model}",
            data_model,
            prior,
            metrics=[
                "reconinfo",
                "reconheur",
                "linregheur",
                "miheur",
            ],
            path=path_to_data,
            num_workers=num_workers,
            seed=seed,
        )
        config.path += "/" + config.name
        config.data_model.coupling = np.unique(
            np.concatenate(
                [np.linspace(0, 0.2, 10), np.linspace(0.2, 0.8, 15)]
            )
        ).tolist()

        config.data_model.length = 1000
        config.prior.size = 500
        config.prior.edge_count = 1250
        config.metrics.reconinfo.num_samples = num_workers
        config.metrics.reconinfo.method = "meanfield"
        config.metrics.reconinfo.start_from_original = False
        config.metrics.reconheur.num_samples = num_workers
        config.metrics.reconheur.method = [
            "transfer_entropy",
            "correlation",
            "granger_causality",
        ]
        config.metrics.linregheur.num_samples = num_workers
        config.metrics.miheur.num_samples = num_workers
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
    config = Figure2HeuristicsConfig.default(
        prior="erdosrenyi",
        data_model="glauber",
        path_to_data="./data",
        num_workers=64,
        seed=None,
        time="12:00:00",
    )
    if args.overwrite and os.path.exists(config.path):
        shutil.rmtree(config.path)
        os.makedirs(config.path)
    path_to_config = f"./configs/{config.name}.pkl"
    config.save(path_to_config)

    script = ScriptManager(
        executable="python ../../midynet/scripts/recon.py",
        execution_command="sbatch",
        path_to_scripts="./scripts",
    )
    args = {
        "run": "Heuristics performance vs recon on erdosrenyi",
        "retrain_data": True,
        "name": config.name,
        # "version": "1.0.0",
        "path_to_config": path_to_config,
        "resume": args.resume,
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
        extra_args=args,
        resources=config.resources.dict,
    )


if __name__ == "__main__":
    main()
