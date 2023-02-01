import numpy as np
import os
import tempfile
import pathlib
import argparse
import shutil

from midynet.config import Config, ExperimentConfig
from midynet.scripts import ScriptManager


class Figure3PredHeuristicsConfig(ExperimentConfig):
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
            f"predheur-{prior}-{data_model}",
            data_model,
            prior,
            metrics=[
                "reconinfo",
                "linregheur",
            ],
            path=path_to_data,
            num_workers=num_workers,
            seed=seed,
        )
        config.path += "/" + config.name
        if data_model == "glauber":
            config.data_model.coupling = np.unique(
                np.concatenate(
                    [np.linspace(0, 0.5, 20), np.linspace(0.5, 2.0, 10)]
                )
            ).tolist()
        elif data_model == "sis":
            config.data_model.infection_prob = np.linspace(0, 0.2, 30).tolist()

        config.data_model.length = 1000
        config.prior.size = 1000
        config.prior.edge_count = 2500
        config.metrics.reconinfo.num_samples = num_workers
        config.metrics.reconinfo.method = "meanfield"
        config.metrics.reconinfo.start_from_original = False
        config.metrics.linregheur.num_samples = num_workers
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
        prior="erdosrenyi",
        data_model="sis",
        path_to_data="./data",
        num_workers=64,
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
        execution_command="sbatch",
        path_to_scripts="./scripts",
    )
    args = {
        "run": "Pred heuristics vs pred on erdosrenyi sis large",
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
        extra_args=args,
        resources=config.resources.dict,
    )


if __name__ == "__main__":
    main()
