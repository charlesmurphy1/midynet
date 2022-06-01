import numpy as np
import itertools
from load_specs import (
    PATH_TO_RUN_EXEC,
    PATH_TO_DATA,
    EXECUTION_COMMAND,
    PATH_TO_LOG,
    SPECS,
)
from midynet.config import ExperimentConfig
from midynet.scripts import ScriptManager


def get_config(
    graph="er", dynamics="sis", num_procs=32, time="24:00:00", mem=12, seed=None
):
    config = ExperimentConfig.default(
        f"recon-{graph}-{dynamics}",
        dynamics,
        graph,
        metrics=["mutualinfo"],
        path=PATH_TO_DATA / "exploration" / f"recon-{graph}-{dynamics}",
        num_procs=num_procs,
        seed=seed,
    )
    N = 100
    E = 250
    T = 100
    if dynamics == "sis":
        coupling = np.linspace(0, 1, 20)
        config.dynamics.set_coupling(coupling)
        config.dynamics.set_value("normalize", False)
        config.dynamics.set_value("recovery_prob", 0.5)
    else:
        coupling = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 4, 10)])
        config.dynamics.set_coupling(coupling)
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.dynamics.set_value("num_steps", T)
    config.metrics.mutualinfo.set_value("num_samples", num_procs)
    config.metrics.mutualinfo.set_value("burn_per_vertex", 5)
    config.metrics.mutualinfo.set_value("start_from_original", False)
    config.metrics.mutualinfo.set_value("initial_burn", 2000)
    config.metrics.mutualinfo.set_value("num_sweeps", 100)
    config.metrics.mutualinfo.set_value("method", "meanfield")

    resources = {
        "account": "def-aallard",
        "time": time,
        "mem": f"{mem}G",
        "cpus-per-task": f"{num_procs}",
        "job-name": f"{config.name}",
    }
    config.insert("resources", resources, force_non_sequence=True, unique=True)
    return config


def main():
    for graph, dynamics in itertools.product(
        ["er", "uniform_cm", "hyperuniform_cm"], ["ising"]
    ):
        config = get_config(graph, dynamics, num_procs=1, time="16:00:00", mem=12)
        script = ScriptManager(
            executable=PATH_TO_RUN_EXEC["run"],
            execution_command=EXECUTION_COMMAND,
            path_to_scripts="./scripts",
            path_to_log=PATH_TO_LOG,
        )
        script.run(
            config,
            resources=config.resources,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=1),
            teardown=False,
        )


if __name__ == "__main__":
    main()
