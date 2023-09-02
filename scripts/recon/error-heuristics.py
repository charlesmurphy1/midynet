import argparse
import os
import pathlib
import shutil
import tempfile
from math import ceil

import numpy as np
from midynet.config import ExperimentConfig, GraphConfig, DataModelConfig
from midynet.scripts import ScriptManager


def format_sequence(*arr):
    arr = [
        np.linspace(*s) if isinstance(s, tuple) and len(s) == 3 else s
        for s in arr
    ]
    return np.unique(np.concatenate(arr)).tolist()


prior_dict = {
    "erdosrenyi": GraphConfig.erdosrenyi(
        size=100, edge_count=250, loopy=False, multigraph=False
    )
}
model_dict = {
    "glauber": DataModelConfig.glauber(
        length=100, coupling=format_sequence((0, 0.8, 25))
    )
}
target_dict = {
    "erdosrenyi": GraphConfig.erdosrenyi(
        size=100, edge_count=250, loopy=True, multigraph=True
    ),
    "littlerock": GraphConfig.littlerock(
        path="/home/murphy9/data/graphs/littlerock.npy",
    ),
}


class ErrorFromHeuristicsConfig:
    @classmethod
    def default(
        cls,
        target="littlerock",
        save_path=None,
        n_workers=1,
        n_samples_per_worker=1,
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
            f"error-heuristics-{target}",
            data_model=model_dict["glauber"],
            prior=prior_dict["erdosrenyi"],
            target=target_dict[target],
            metrics=["recon_error"],
            path=save_path,
            n_workers=n_workers,
            n_async_jobs=n_async_jobs,
            seed=seed,
        )

        config.metrics.recon_error.n_samples = n_samples_per_worker * ceil(
            n_workers / n_async_jobs
        )
        config.metrics.recon_error.reconstructor = [
            "bayesian",
            "correlation",
            "granger_causality",
            "transfer_entropy",
        ]
        config.metrics.recon_error.data_mcmc.n_sweeps = 1000
        config.metrics.recon_error.measures = (
            "roc, posterior_similarity, accuracy, error_prob"
        )

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
    for target in ["littlerock"]:
        config = ErrorFromHeuristicsConfig.default(
            target,
            n_workers=64,
            n_samples_per_worker=1,
            n_async_jobs=2,
            time="24:00:00",
            mem=16,
            save_path=f"/home/murphy9/data/error-heuristics/{target}",
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
                "mpi4py",
            ],
            virtualenv="/home/murphy9/.midynet-env/bin/activate",
            extra_args=args,
            resources=config.resources.dict,
        )


if __name__ == "__main__":
    main()
