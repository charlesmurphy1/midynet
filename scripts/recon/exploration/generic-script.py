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

        config.resources = dict(
            account="def-aallard",
            time=time,
            mem=f"{mem}G",
            cpus_per_task=num_procs,
            job_name=config.name,
        )
        config.lock()
        return config


def main():

    config = TestingConfig.default(
        prior="erdosrenyi",
        data_model="sis",
        path_to_data="./testing",
        num_procs=1,
        time="24:00:00",
        mem=12,
        seed=None,
    )
    script = ScriptManager(
        executable="python ../../../midynet/scripts/run_reconstruction.py",
        execution_command="bash",
        path_to_scripts="./scripts",
        path_to_log="./log",
    )
    script.run(
        config,
        run_name="local testing for recon-erdos-sis",
        modules_to_load=[
            "StdEnv/2020",
            "gcc/9",
            "python/3.8",
            "graph-tool",
            "scipy-stack",
        ],
        virtualenv="None",
    )


if __name__ == "__main__":
    main()
