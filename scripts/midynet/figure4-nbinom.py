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
    dynamics="sis",
    num_procs=4,
    num_async_process=1,
    time="24:00:00",
    mem=12,
    seed=None,
):
    config = ExperimentConfig.default(
        f"figure4-nbinom-{dynamics}",
        dynamics,
        "nbinom_cm",
        metrics=["mutualinfo"],
        path=PATH_TO_DATA / "figure4",
        seed=seed,
        num_procs=num_procs,
        num_async_process=num_async_process,
    )
    N, E = 100, 250
    T = [100]
    h = [0, 0.5, 1]
    if dynamics == "sis":
        coupling = np.unique(
            np.concatenate(
                [np.linspace(0, 1., 20), np.linspace(1., 2, 20)]
            )
        )
    elif dynamics == "ising":
        coupling = np.unique(
            np.concatenate([np.linspace(0, 1, 20), np.linspace(1, 4, 20)])
        )
    elif dynamics == "cowan":
        coupling = np.unique(
            np.concatenate([np.linspace(1, 2, 20), np.linspace(2, 6, 20)])
        )
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.graph.set_value("heterogeneity", h)
    config.dynamics.set_value("num_steps", T)
    config.dynamics.set_coupling(coupling)
    if dynamics == "sis":
        config.dynamics.set_value("recovery_prob", 0.5)
    if dynamics == "sis" or dynamics == "cowan":
        config.dynamics.set_value("auto_activation_prob", 0.001)
    config.metrics.mutualinfo.set_value("num_samples", num_procs)
    config.metrics.mutualinfo.set_value("method", "full-meanfield")
    config.metrics.mutualinfo.set_value("initial_burn", 2000)
    config.metrics.mutualinfo.set_value("num_sweeps", 1000)
    config.metrics.mutualinfo.set_value("burn_per_vertex", 5)

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
        config = get_config(
            dynamics, num_procs=40, mem=12, time="8:00:00"
        )
        script = ScriptManager(
            executable=PATH_TO_RUN_EXEC["run"],
            execution_command=EXECUTION_COMMAND,
            path_to_scripts="./scripts",
        )

        configT = script.split_param(config, "dynamics.num_steps")

        script.run(
            configT,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
            teardown=False,
        )


if __name__ == "__main__":
    main()
