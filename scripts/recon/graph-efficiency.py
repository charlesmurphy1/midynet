import argparse
import os
import pathlib
import shutil
import tempfile

import numpy as np
from midynet.config import (
    DataModelConfig,
    ExperimentConfig,
    GraphConfig,
    MetricsConfig,
)
from midynet.scripts import ScriptManager


def format_sequence(*arr):
    arr = [
        np.linspace(*s) if isinstance(s, tuple) and len(s) == 3 else s
        for s in arr
    ]
    return np.unique(np.concatenate(arr)).tolist()


data_models = {
    "erdosrenyi": DataModelConfig.glauber(length=2000, coupling=0.2),
    "planted_partition": DataModelConfig.glauber(length=2000, coupling=0.2),
    "littlerock": DataModelConfig.glauber(length=2000, coupling=0.03),
    "polblogs": DataModelConfig.glauber(length=2000, coupling=0.013),
}
priors = {
    "erdosrenyi": GraphConfig.erdosrenyi(
        size=100, edge_count=250, loopy=True, multigraph=True
    ),
    "configuration": GraphConfig.configuration(100, 250),
    "degree_corrected_stochastic_block_model": GraphConfig.degree_corrected_stochastic_block_model(
        size=100,
        edge_count=250,
    ),
    "stochastic_block_model": GraphConfig.stochastic_block_model(
        size=100, edge_count=250, loopy=True, multigraph=True
    ),
}
targets = {
    "erdosrenyi": GraphConfig.erdosrenyi(size=100, edge_count=250),
    "planted_partition": GraphConfig.planted_partition(
        size=100, edge_count=250, block_count=3, loopy=True, multigraph=True
    ),
    "food_web": GraphConfig.littlerock(
        # path="../data/graphs/littlerock.npy"
        path="/home/murphy9/data/graphs/littlerock.npy"
    ),
    "polblogs": GraphConfig.polblogs(
        # path="../data/graphs/polblogs.npy",
        path="/home/murphy9/data/graphs/polblogs.npy",
    ),
}


class EfficiencyGraphsConfig:
    @classmethod
    def default(
        cls,
        target,
        path_to_data=None,
        n_workers=1,
        n_samples_per_worker=1,
        time="24:00:00",
        mem=12,
        seed=None,
    ):
        path_to_data = pathlib.Path(
            tempfile.mktemp() if path_to_data is None else path_to_data
        )
        if not os.path.exists(path_to_data):
            os.makedirs(path_to_data)
        metrics = [
            MetricsConfig.efficiency(
                graph_mcmc="meanfield", data_mcmc="meanfield"
            )
        ]

        config = ExperimentConfig.default(
            f"{target}-glauber",
            "glauber",
            "erdosrenyi",
            target=target,
            metrics=metrics,
            path=path_to_data,
            n_workers=n_workers,
            seed=seed,
        )

        config.data_model = data_models[target]
        config.prior = list(priors.values())
        # config.prior = [priors["stochastic_block_model"]]
        config.metrics.efficiency.n_samples = n_samples_per_worker * n_workers
        # config.metrics.efficiency.resample_graph = target != "polblogs"
        config.metrics.efficiency.data_mcmc.n_sweeps = 1000
        config.metrics.efficiency.data_mcmc.n_steps_per_vertex = 10
        config.metrics.efficiency.data_mcmc.burn = 2000
        config.metrics.efficiency.data_mcmc.graph_rate = 1
        config.metrics.efficiency.data_mcmc.prior_rate = 1
        config.metrics.efficiency.data_mcmc.param_rate = 0
        config.metrics.efficiency.data_mcmc.start_from_original = True
        if config.metrics.efficiency.graph_mcmc is not None:
            config.metrics.efficiency.graph_mcmc.n_sweeps = 1000
            config.metrics.efficiency.graph_mcmc.burn = 2000
        config.metrics.efficiency.reduction = "identity"
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

    for model in [
        "erdosrenyi",
        "planted_partition",
        "littlerock",
        "polblogs",
    ]:
        config = EfficiencyGraphsConfig.default(
            model,
            n_workers=64,
            n_samples_per_worker=1,
            time="12:00:00",
            mem=64,
            # path_to_data=f"./tests/recon-{model}",
            path_to_data=f"/home/murphy9/data/graph-efficiency/recon-{model}",
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
        extra_args = {
            "run": f"graph-eff with {model}",
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
