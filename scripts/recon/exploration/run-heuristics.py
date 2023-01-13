import numpy as np
import os
import tempfile
import pathlib
import argparse
import shutil

from midynet.config import Config, ExperimentConfig
from midynet.scripts import ScriptManager


class HeuristicsConfig(ExperimentConfig):
    @classmethod
    def default(
        cls,
        prior="erdosrenyi",
        target=None,
        data_model="glauber",
        path_to_data=None,
        num_procs=1,
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
            target=target,
            metrics=["reconinfo", "reconheuristics"],
            path=path_to_data,
            num_procs=num_procs,
            seed=seed,
        )
        # config.data_model.infection_prob = [0.0, 0.1, 0.5, 1.0]
        config.data_model.coupling = np.linspace(0, 3, 20).tolist()
        config.prior.size = 100
        config.prior.edge_count = 250
        config.prior.with_self_loops = (
            config.prior.with_parallel_edges
        ) = False
        config.data_model.length = 100
        config.metrics.reconinfo.num_samples = 2 * num_procs
        config.metrics.reconinfo.method = "meanfield"
        config.metrics.reconheuristics.num_samples = 2 * num_procs
        config.metrics.reconheuristics.method = [
            "transfer_entropy",
            "correlation",
            "granger_causality",
        ]
        config.resources.update(
            account="def-aallard",
            time=time,
            mem=mem,
            cpus_per_task=config.num_procs,
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
    config = HeuristicsConfig.default(
        prior="erdosrenyi",
        data_model="glauber",
        path_to_data="./heur-er-glauber",
        num_procs=12,
        seed=None,
    )
    if args.overwrite and os.path.exists(config.path):
        shutil.rmtree(config.path)
        os.makedirs(config.path)
    path_to_config = f"./configs/{config.name}.pkl"
    config.save(path_to_config)

    script = ScriptManager(
        executable="python ../../../midynet/scripts/recon.py",
        execution_command="bash",
        path_to_scripts="./scripts",
    )
    args = {
        "run": "Heuristics performance vs recon on erdos-glauber",
        "name": "heur-er-glauber",
        "version": "1.0.0",
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
        ],
        virtualenv=None,
        extra_args=args,
        resources=config.resources.dict,
    )


if __name__ == "__main__":
    main()
