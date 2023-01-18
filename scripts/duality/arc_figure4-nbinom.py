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
        f"figure4-large-nbinom-{dynamics}",
        dynamics,
        "nbinom_cm",
        metrics=["mutualinfo"],
        path=PATH_TO_DATA / "figure4_" / f"figure4-large-nbinom-{dynamics}",
        seed=seed,
        num_procs=num_procs,
        num_async_process=num_async_process,
    )
    N, E = 1000, 2500
    T = 2000
    h = 1
    if dynamics == "sis":
        coupling = np.unique(
            np.concatenate(
                [
                    np.linspace(0., 0.125, 3),
                    np.linspace(0.125, 0.5, 30),
                ]
            )
        )
    elif dynamics == "ising":
        coupling = np.unique(
            np.concatenate([np.linspace(0., 0.5, 6), np.linspace(0.5, 0.75, 22), np.linspace(0.75, 1.5, 6)])
        )
    elif dynamics == "cowan_backward":
        coupling = np.unique(
            np.concatenate(
                [
                    np.linspace(1.0, 1.25, 4),
                    np.linspace(1.25, 1.5, 20),
                    np.linspace(1.5, 3, 10),
                ]
            )
        )
    elif dynamics == "cowan_forward" or dynamics == "cowan":
        coupling = np.unique(
            np.concatenate(
                [
                    np.linspace(2., 2.5, 23),
                    np.linspace(2.5, 4, 10),
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
        config.dynamics.set_value("auto_activation_prob", 0.001)
        config.dynamics.set_value("num_active", 1)
    config.metrics.mutualinfo.set_value(
        "num_samples", num_procs // num_async_process
    )
    config.metrics.mutualinfo.set_value("method", "meanfield")
    config.metrics.mutualinfo.set_value("initial_burn", 10000)
    config.metrics.mutualinfo.set_value("num_sweeps", 500)
    config.metrics.mutualinfo.set_value("burn_per_vertex", 5)
    if dynamics == "cowan_backward":
        config.dynamics.set_value("num_active", config.graph.size)
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
    for dynamics in ["ising", "sis", "cowan_forward", "cowan_backward"]:
        config = get_config(
            dynamics,
            num_procs=64,
            num_async_process=2,
            mem=18,
            time="32:00:00",
        )
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
