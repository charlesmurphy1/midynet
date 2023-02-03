import numpy as np
import pathlib
import tempfile
import argparse
import os
import shutil
from midynet.config import ExperimentConfig
from midynet.scripts import ScriptManager


class Figure2ExactConfig(ExperimentConfig):
    @classmethod
    def default(
        cls,
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
            f"exact-erdosrenyi-{data_model}",
            data_model,
            "erdosrenyi",
            metrics=[
                "reconinfo",
                "linregheur",
            ],
            path=path_to_data,
            num_workers=num_workers,
            seed=seed,
        )
        N = 5
        E = 5
        T = np.unique(np.logspace(1, 4, 100).astype("int")).tolist()
        if data_model == "sis":
            config.data_model.recovery_prob = 0.1
            config.data_model.infection_prob = [0.25, 0.5, 1]
            config.data_model.auto_infection_prob = 1e-4
        elif data_model == "glauber":
            config.data_model.coupling = [0.25, 0.5, 1]
        elif data_model == "cowan":
            config.data_model.nu = [0.5, 1, 2]
            config.data_model.eta = 0.1
        config.data_model.length = T

        config.prior.size = N
        config.prior.edge_count = E
        config.prior.with_self_loops = False
        config.prior.with_parallel_edges = False
        config.metrics.reconinfo.num_samples = 100 * num_workers
        config.metrics.reconinfo.method = "exact"
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
    for data_model in ["glauber", "sis", "cowan"]:
        config = Figure2ExactConfig.default(
            data_model, num_procs=40, time="24:00:00", mem=12
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
