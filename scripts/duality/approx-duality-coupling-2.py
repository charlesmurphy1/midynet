import numpy as np
import pathlib
import tempfile
import argparse
import os
import shutil
from midynet.config import ExperimentConfig, DataModelConfig, MetricsConfig
from midynet.scripts import ScriptManager

def format_sequence(*arr):
    arr = [np.linspace(*s) if isinstance(s, tuple) and len(s) == 3 else s for s in arr]
    return np.unique(np.concatenate(arr)).tolist()
    

couplings =  {
    "glauber": format_sequence((0, 0.02, 5), (0.02, 0.04, 20), (0.04, 0.1, 10)),
    # "sis": format_sequence((0, 0.02, 20), (0.02, 0.06, 10)),
    "sis": format_sequence((0.01, 0.02, 20)),
    "cowan_forward": format_sequence((0, 0.07, 5), (0.07, 0.2, 30), (0.2, 0.3, 5)),
    "cowan_backward": format_sequence((0, 0.1, 20), (0.1, 0.3, 10)),

}
STEP_FACTOR = 4

model_dict = {
    "littlerock": DataModelConfig.glauber(length=2000, coupling=couplings["glauber"]),
    "euairlines": DataModelConfig.sis(length=2 * 450, infection_prob=couplings["sis"], recovery_prob=0.5),
    "celegans_forward": DataModelConfig.cowan_forward(length=STEP_FACTOR * 514, nu=couplings["cowan_forward"]),
    "celegans_backward": DataModelConfig.cowan_backward(length=STEP_FACTOR * 514, nu=couplings["cowan_backward"]),
}

paths =  {
    "littlerock": None,
    "euairlines": None,
    "celegans_forward": None,
    "celegans_backward": None,
}

class Figure4CMRealNetworkConfig:
    @classmethod
    def default(
        cls,
        target,
        path_to_data=None,
        n_workers=1,
        time="24:00:00",
        mem=12,
        seed=None,
    ):

        path_to_data = pathlib.Path(
            tempfile.mktemp() if path_to_data is None else path_to_data
        )
        if not os.path.exists(path_to_data):
            os.makedirs(path_to_data)
        metrics = [MetricsConfig.efficiency(graph_mcmc=None, data_mcmc="meanfield")]
        config = ExperimentConfig.default(
            f"{target}",
            model_dict[target],
            "degree_constrained_configuration",
            target=target,
            metrics=metrics,
            path=path_to_data,
            n_workers=n_workers,
            seed=seed,
            target_params=dict(path=paths[target])
        )

        config.metrics.efficiency.n_samples = n_workers
        config.metrics.efficiency.resample_graph = True
        config.metrics.efficiency.data_mcmc.n_sweeps = 200
        config.metrics.efficiency.data_mcmc.start_from_original = True
        config.metrics.efficiency.data_mcmc.burn = 2000
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
    # for target in model_dict.keys():
    for target in ["euairlines"]:
        config = Figure4CMRealNetworkConfig.default(
            target, n_workers=4, time="24:00:00", mem=12
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
        extra_args = {
            "run": "Measures on CM with real graphs",
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
        break


if __name__ == "__main__":
    main()
