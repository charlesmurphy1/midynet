import argparse
import os
import pathlib
import shutil
import tempfile
from math import ceil

import numpy as np
from midynet.config import ExperimentConfig
from midynet.scripts import ScriptManager


def format_sequence(*arr):
    arr = [
        np.linspace(*s) if isinstance(s, tuple) and len(s) == 3 else s
        for s in arr
    ]
    return np.unique(np.concatenate(arr)).tolist()


class ErrorFromHeuristicsConfig:
    @classmethod
    def default(
        cls,
        reconstructor="bayesian",
        save_path=None,
        n_workers=1,
        n_async_jobs=1,
        time="24:00:00",
        mem=12,
        seed=None,
    ):

        save_path = pathlib.Path(
            tempfile.mktemp() if save_path is None else save_path
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        config = ExperimentConfig.default(
            f"error-{reconstructor}",
            "glauber",
            "erdosrenyi",
            metrics=["recon_error"],
            path=save_path,
            n_workers=n_workers,
            n_async_jobs=n_async_jobs,
            seed=seed,
        )
        N, M, T = 5, 5, 100
        config.prior.size = N
        config.prior.edge_count = M
        # config.prior.loopy = False
        # config.prior.multigraph = False
        config.data_model.length = T
        config.data_model.coupling = format_sequence((0, 0.8, 25))

        config.metrics.recon_error.n_samples = 10 * ceil(
            n_workers / n_async_jobs
        )
        config.metrics.recon_error.reconstructor = reconstructor
        config.metrics.recon_error.data_mcmc.n_sweeps = 1000
        config.metrics.recon_error.data_mcmc.method = "exact"

        # config.metrics.reconinfo.n_samples = ceil(n_workers / n_async_jobs)
        # config.metrics.reconinfo.data_mcmc.method = "exact"
        # config.metrics.reconinfo.data_mcmc.n_sweeps = 1000
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
    for reconstructor in ["bayesian", "correlation", "granger_causality"]:
        config = ErrorFromHeuristicsConfig.default(
            reconstructor,
            n_workers=4,
            time="24:00:00",
            mem=12,
            save_path=f"../data/error-heuristics/{reconstructor}",
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
