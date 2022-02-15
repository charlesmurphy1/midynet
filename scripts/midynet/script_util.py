import json
import os
import pathlib
import numpy as np

from midynet.config import ExperimentConfig

SPECS = json.load(open("specs.json", "r"))
if os.getenv("SERVER_NAME") in SPECS:
    SPECS = SPECS[os.getenv("SERVER_NAME")]
else:
    SPECS = SPECS["default"]

PATH_TO_DATA = pathlib.Path(SPECS["path_to_data"])
PATH_TO_RUN_EXEC = dict(run="python ../../midynet/scripts/run.py")
PATH_TO_LOG = pathlib.Path("./log")
if not PATH_TO_LOG.exists():
    PATH_TO_LOG.mkdir()
EXECUTION_COMMAND = SPECS["command"]


def get_config_test(num_procs=4, time="1:00:00", mem=12, seed=1):
    config = ExperimentConfig.default(
        "test",
        "sis",
        "ser",
        metrics=["mutualinfo", "graph_entropy"],
        path=PATH_TO_DATA / "test",
        num_procs=num_procs,
        seed=seed,
    )
    N, E, T = 5, 5, 5
    coupling = np.linspace(0, 3, 10)
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.dynamics.set_value("num_steps", T)
    config.dynamics.set_coupling(coupling)
    config.metrics.mutualinfo.set_value("num_samples", 1000)
    config.metrics.mutualinfo.set_value("method", "exact")
    resources = {
        "account": "def-aallard",
        "time": time,
        "mem": f"{mem}G",
        "cpus-per-task": f"{num_procs}",
        "job-name": f"{config.name}",
        "output": PATH_TO_LOG / f"{config.name}.out",
    }
    config.insert("resources", resources, force_non_sequence=True, unique=True)
    return config


def get_config_figure4Nbinom(
    dynamics="sis", num_procs=4, time="24:00:00", mem=12, seed=None
):
    config = ExperimentConfig.default(
        f"figure4-nbinom-{dynamics}",
        dynamics,
        "nbinom_cm",
        metrics=["mutualinfo"],
        path=PATH_TO_DATA / "figure4",
        seed=seed,
        num_procs=num_procs,
    )
    N, E = 100, 250
    T = [10, 100]
    h = np.linspace(0.001, 5, 20)
    if dynamics == "sis" or dynamics == "ising":
        coupling = np.unique(
            np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 4, 10)])
        )
    elif dynamics == "cowan":
        coupling = np.unique(
            np.concatenate(
                [
                    np.linspace(0, 1, 5),
                    np.linspace(1, 2, 10),
                    np.linspace(2, 4, 5),
                ]
            )
        )
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.graph.set_value("heterogeneity", h)
    config.dynamics.set_value("num_steps", T)
    config.dynamics.set_coupling(coupling)
    config.metrics.mutualinfo.set_value("num_samples", 32)
    config.metrics.mutualinfo.set_value("method", "meanfield")
    config.metrics.mutualinfo.set_value("num_sweeps", 200)

    resources = {
        "account": "def-aallard",
        "time": time,
        "mem": f"{mem}G",
        "cpus-per-task": f"{num_procs}",
        "job-name": f"{config.name}",
        "output": PATH_TO_LOG / f"{config.name}.out",
    }
    config.insert("resources", resources, force_non_sequence=True, unique=True)
    return config


if __name__ == "__main__":
    pass
