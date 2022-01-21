import json
import numpy as np
import os
import pathlib
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


def config_figure2Exact(
    dynamics="sis", num_procs=4, time="24:00:00", mem=12, seed=None
):
    config = ExperimentConfig.default(
        "figure2-exact",
        dynamics,
        "ser",
        metrics=["mutualinfo"],
        path=PATH_TO_DATA / "figure2" / "exact",
        num_procs=num_procs,
        seed=seed,
    )
    N = 5
    E = np.arange(1, int(N * (N - 1) / 2))
    T = [10, 25, 50, 100, 250, 500, 1000]
    coupling = np.linspace(0, 3, 30)
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.dynamics.set_value("num_steps", T)
    config.dynamics.set_coupling(coupling)
    config.metrics.mutualinfo.set_value("num_samples", 500 // num_procs * num_procs)
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


def config_figure2MCMC(dynamics="sis", num_procs=4, time="24:00:00", mem=12, seed=None):
    config = ExperimentConfig.default(
        "figure2-mcmc",
        dynamics,
        "ser",
        metrics=["mutualinfo"],
        path=PATH_TO_DATA / "figure2" / "mcmc",
        seed=seed,
        num_procs=num_procs,
    )
    N, E = 5, 5
    T = 10
    coupling = np.linspace(0, 3, 30)
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.dynamics.set_value("num_steps", T)
    config.dynamics.set_coupling(coupling)
    config.metrics.mutualinfo.set_value("num_samples", 100 // num_procs * num_procs)
    config.metrics.mutualinfo.set_value("method", ["meanfield", "annealed"])
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


def config_figure2Large(
    dynamics="sis", num_procs=4, time="24:00:00", mem=12, seed=None
):
    config = ExperimentConfig.default(
        "figure2-large",
        dynamics,
        "ser",
        metrics=["mutualinfo"],
        path=PATH_TO_DATA / "figure2" / "large",
        seed=seed,
        num_procs=num_procs,
    )
    N, E = 100, 250
    T = 100
    coupling = np.linspace(0, 3, 30)
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.dynamics.set_value("num_steps", T)
    config.dynamics.set_coupling(coupling)
    config.metrics.mutualinfo.set_value("num_samples", 100 // num_procs * num_procs)
    config.metrics.mutualinfo.set_value("method", ["meanfield", "annealed"])

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
