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
    config.metrics.mutualinfo.set_value("num_samples", 1000 // num_procs * num_procs)
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


def get_config_figure2Exact(
    dynamics="sis", num_procs=4, time="24:00:00", mem=12, seed=None
):
    config = ExperimentConfig.default(
        f"figure2-exact-{dynamics}",
        dynamics,
        "ser",
        metrics=["mutualinfo"],
        path=PATH_TO_DATA / "figure2-test" / f"exact-{dynamics}",
        num_procs=num_procs,
        seed=seed,
    )
    N = 5
    E = np.arange(1, int(N * (N - 1) / 2))
    T = [10, 25, 50, 100, 250, 500, 1000]
    if dynamics == "sis":
        coupling = np.linspace(0, 1, 100)
        config.dynamics.set_coupling(coupling)
        config.dynamics.set_value("normalize", False)
    else:
        coupling = np.linspace(0, 4, 100)
        config.dynamics.set_coupling(coupling)
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.dynamics.set_value("num_steps", T)
    config.metrics.mutualinfo.set_value(
        "num_samples", max(1, 1000 // num_procs) * num_procs
    )
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


def get_config_figure3Small(
    dynamics="sis", num_procs=4, time="24:00:00", mem=12, seed=None
):
    config = ExperimentConfig.default(
        f"figure3-small-{dynamics}",
        dynamics,
        "ser",
        metrics=["mutualinfo"],
        path=PATH_TO_DATA / "figure3" / "small",
        seed=seed,
        num_procs=num_procs,
    )
    N, E = 5, 5
    T = 10

    if dynamics == "sis":
        coupling = np.linspace(0, 1, 30)
        config.dynamics.set_coupling(coupling)
        config.dynamics.set_value("normalize", False)
    else:
        coupling = np.linspace(0, 4, 30)
        config.dynamics.set_coupling(coupling)
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.dynamics.set_value("num_steps", T)
    config.metrics.mutualinfo.set_value(
        "num_samples", max(1, 1000 // num_procs) * num_procs
    )
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


def get_config_figure3Large(
    dynamics="sis", num_procs=4, time="24:00:00", mem=12, seed=None
):
    config = ExperimentConfig.default(
        f"figure3-large-{dynamics}",
        dynamics,
        "ser",
        metrics=["mutualinfo"],
        path=PATH_TO_DATA / "figure3" / "large",
        seed=seed,
        num_procs=num_procs,
    )
    N, E = 100, 250
    T = 100
    if dynamics == "sis":
        coupling = np.linspace(0, 1, 30)
        config.dynamics.set_coupling(coupling)
        config.dynamics.set_value("normalize", False)
    else:
        coupling = np.linspace(0, 4, 30)
        config.dynamics.set_coupling(coupling)
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.dynamics.set_value("num_steps", T)
    config.metrics.mutualinfo.set_value(
        "num_samples", max(1, 100 // num_procs) * num_procs
    )
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


def get_config_figure4Nbinom(
    dynamics="sis", num_procs=4, time="48:00:00", mem=12, seed=None
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
    T = [10, 100, 1000]
    h = np.linspace(0.001, 5, 40)
    if dynamics == "sis" or dynamics == "ising":
        coupling = np.unique(
            np.concatenate([np.linspace(0, 1, 20), np.linspace(1, 4, 20)])
        )
    elif dynamics == "cowan":
        coupling = np.unique(
            np.concatenate(
                [np.linspace(0, 1, 10), np.linspace(1, 2, 20), np.linspace(2, 4, 10)]
            )
        )
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.graph.set_value("heterogeneity", h)
    config.dynamics.set_value("num_steps", T)
    config.dynamics.set_coupling(coupling)
    config.metrics.mutualinfo.set_value(
        "num_samples", max(1, 25 // num_procs) * num_procs
    )
    config.metrics.mutualinfo.set_value("method", "meanfield")

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
