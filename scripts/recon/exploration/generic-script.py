import numpy as np
import os
import tempfile
import pathlib
import argparse

from midynet.config import Config, ExperimentConfig, frozen
from midynet.scripts import ScriptManager


class TestingConfig(ExperimentConfig):
    @classmethod
    def default(
        cls,
        prior="erdosrenyi",
        data_model="sis",
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
            f"recon-{prior}-{data_model}",
            data_model,
            prior,
            metrics=["recon_information"],
            path=path_to_data,
            num_procs=num_procs,
            seed=seed,
        )
        config.data_model.infection_prob = [0.0, 0.1, 0.5, 1.0]
        config.prior.size = 5
        config.prior.edge_count = 5
        config.data_model.length = 100
        config.metrics.recon_information.num_samples = num_procs
        config.metrics.recon_information.burn_per_vertex = 5
        config.metrics.recon_information.start_from_original = False
        config.metrics.recon_information.initial_burn = 2000
        config.metrics.recon_information.num_sweeps = 100
        config.metrics.recon_information.method = "exact"
        config.lock()
        return config


def main():

    if not os.path.exists("./configs"):
        os.mkdir("./configs")
    if not os.path.exists("./log"):
        os.mkdir("./log")
    config = TestingConfig.default(
        prior="erdosrenyi",
        data_model="sis",
        path_to_data="./testing",
        num_procs=1,
        seed=None,
    )
    resources = {
        "account": "def-aallard",
        "time": "24:00:00",
        "mem": "12G",
        "cpus-per-task": config.num_procs,
        "job-name": config.name,
        "output": f"log/{config.name}.out",
    }
    path_to_config = f"./configs/{config.name}.pkl"
    config.save(path_to_config)

    script = ScriptManager(
        executable="python ../../../midynet/scripts/run_reconstruction.py",
        execution_command="bash",
        path_to_scripts="./scripts",
    )
    args = {
        # "run_name": "local testing for recon-erdos-sis",
        "path_to_config": path_to_config,
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
        resources=resources,
    )


if __name__ == "__main__":
    main()
