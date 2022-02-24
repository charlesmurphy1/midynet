import numpy as np

from midynet.config import ExperimentConfig
from midynet.scripts import ScriptManager

from load_specs import (
    PATH_TO_DATA,
    PATH_TO_LOG,
    PATH_TO_RUN_EXEC,
    EXECUTION_COMMAND,
    SPECS,
)


def get_config(
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
    T = [500]
    h = np.linspace(0.001, 5, 40)
    if dynamics == "sis" or dynamics == "ising":
        coupling = np.unique(
            np.concatenate([np.linspace(0, 1, 20), np.linspace(1, 4, 20)])
        )
    elif dynamics == "cowan":
        coupling = np.unique(
            np.concatenate(
                [
                    np.linspace(0, 1, 10),
                    np.linspace(1, 2, 20),
                    np.linspace(2, 4, 10),
                ]
            )
        )
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.graph.set_value("heterogeneity", h)
    config.dynamics.set_value("num_steps", T)
    config.dynamics.set_coupling(coupling)
    if dynamics == "sis":
        config.dynamics.set_value("recovery_prob", 0.5)
    config.metrics.mutualinfo.set_value("num_samples", num_procs)
    config.metrics.mutualinfo.set_value("method", "meanfield")
    config.metrics.mutualinfo.set_value("num_sweeps", 1000)
    config.metrics.mutualinfo.set_value("burn_per_vertex", 1)

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


def main():
    for dynamics in ["ising", "sis", "cowan"]:
        config = get_config(dynamics, num_procs=64, mem=12, time="40:00:00")
        script = ScriptManager(
            executable=PATH_TO_RUN_EXEC["run"],
            execution_command=EXECUTION_COMMAND,
            path_to_scripts="./scripts",
        )

        script.run(
            config,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
            teardown=False,
        )


if __name__ == "__main__":
    main()
