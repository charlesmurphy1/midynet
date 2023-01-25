import numpy as np
import os
import tempfile
import pathlib
import argparse
import shutil

from midynet.config import Config, ExperimentConfig
from midynet.scripts import ScriptManager


class LittleRockModelSelectionConfig(ExperimentConfig):
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
            f"test-{prior}-{data_model}",
            data_model,
            prior,
            target="littlerock",
            metrics=[
                "reconinfo",
                "targreconinfo",
            ],
            path=path_to_data,
            num_workers=num_workers,
            seed=seed,
        )
        config.path += "/" + config.name
        config.data_model.coupling = 0.05
        config.data_model.length = 2000
        config.metrics.targreconinfo.num_samples = 4 * num_workers
        config.metrics.targreconinfo.method = "meanfield"
        config.metrics.targreconinfo.start_from_original = False
        config.metrics.targreconinfo.num_sweeps = 1000
        config.metrics.targreconinfo.reduction = "identity"
        config.metrics.reconinfo.num_samples = 4 * num_workers
        config.metrics.reconinfo.method = "meanfield"
        config.metrics.reconinfo.start_from_original = False
        config.metrics.reconinfo.num_sweeps = 1000
        config.metrics.reconinfo.reduction = "identity"
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
    for prior in ["erdosrenyi", "configuration"]:
        config = LittleRockModelSelectionConfig.default(
            prior=prior,
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
            executable="python ../../../midynet/scripts/recon.py",
            execution_command="sbatch",
            path_to_scripts="./scripts",
        )
        extra_args = {
            "run": f"Netsci figure for model selection {prior}-glauber",
            "push_data": False,
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
            extra_args=extra_args,
            resources=config.resources.dict,
        )


if __name__ == "__main__":
    main()
