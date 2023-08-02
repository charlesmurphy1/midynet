import argparse
import os
import pathlib
import shutil
import tempfile

import numpy as np
from math import ceil
from midynet.config import DataModelConfig, ExperimentConfig, MetricsConfig
from midynet.scripts import ScriptManager


def format_sequence(*arr):
    arr = [
        np.linspace(*s) if isinstance(s, tuple) and len(s) == 3 else s
        for s in arr
    ]
    return np.unique(np.concatenate(arr)).tolist()


couplings = {
    "glauber": format_sequence((0, 0.02, 5), (0.02, 0.04, 25), (0.04, 0.1, 5)),
    "sis": format_sequence((0, 0.02, 20), (0.02, 0.5, 30)),
    "cowan_forward": format_sequence(
        (0, 0.07, 5), (0.07, 0.2, 30), (0.2, 0.3, 5)
    ),
    "cowan_backward": format_sequence((0, 0.1, 25), (0.1, 0.3, 15)),
}
STEP_FACTOR = 4

graph_dict = {
    "glauber": ("littlerock", "/home/murphy9/data/graphs/littlerock.pkl"),
    # "glauber": ("polblogs", "/home/murphy9/data/graphs/polblogs.pkl"),
    "sis": ("euairlines", "/home/murphy9/data/graphs/euairlines.pkl"),
    # "sis": ("euairlines", "../../data/graphs/euairlines.pkl"),
    "cowan_forward": ("celegans", "/home/murphy9/data/graphs/celegans.pkl"),
    "cowan_backward": ("celegans", "/home/murphy9/data/graphs/celegans.pkl"),
}

model_dict = {
    "glauber": DataModelConfig.glauber(
        length=2000, coupling=couplings["glauber"]
    ),
    "sis": DataModelConfig.sis(
        length=2000, infection_prob=couplings["sis"], recovery_prob=0.5
    ),
    "cowan_forward": DataModelConfig.cowan_forward(
        length=5000, nu=couplings["cowan_forward"]
    ),
    "cowan_backward": DataModelConfig.cowan_backward(
        length=5000, nu=couplings["cowan_backward"]
    ),
}


class Figure4CMRealNetworkConfig:
    @classmethod
    def default(
        cls,
        model,
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
        metrics = [
            MetricsConfig.reconinfo(graph_mcmc=None, data_mcmc="meanfield")
        ]
        target, target_path = graph_dict[model]
        assert os.path.exists(
            target_path
        ), f"path {target_path} does not exist."
        assert os.path.exists(
            path_to_data
        ), f"path {path_to_data} does not exist."
        config = ExperimentConfig.default(
            f"{target}-{model}",
            model_dict[model],
            "degree_constrained_configuration",
            target=target,
            metrics=metrics,
            path=path_to_data,
            n_workers=n_workers,
            n_async_jobs=n_async_jobs,
            seed=seed,
            target_params=dict(path=target_path),
        )

        config.prior.size = config.target.size
        if "backward" in model:
            config.data_model.n_active = config.prior.size
        elif model == "glauber":
            config.data_model.n_active = -1
        else:
            config.data_model.n_active = ceil(0.01 * config.prior.size)
        config.metrics.reconinfo.n_samples = n_workers // n_async_jobs
        config.metrics.reconinfo.data_mcmc.n_sweeps = 1000
        config.metrics.reconinfo.data_mcmc.n_steps_per_vertex = 1
        config.metrics.reconinfo.data_mcmc.n_gibbs_sweeps = 5
        config.metrics.reconinfo.data_mcmc.sample_prior = False
        # config.metrics.reconinfo.data_mcmc.start_from_original = True
        config.metrics.reconinfo.data_mcmc.burn_sweeps = 5
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
    # for model in model_dict.keys():
    for model in [ "cowan_forward", "cowan_backward"]:
        config = Figure4CMRealNetworkConfig.default(
            model,
            n_workers=64,
            n_async_jobs=4,
            time="48:00:00",
            mem=0,
            path_to_data=f"/home/murphy9/data/midynet/duality-coupling/{model}-{graph_dict[model][0]}",
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
            "run": f"Measures on CM with real graphs - {model} - tr2",
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
