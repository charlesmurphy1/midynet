import numpy as np
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
    dynamics="sis", num_procs=32, time="24:00:00", mem=12, seed=None
):
    config = ExperimentConfig.default(
        f"figure2-exact-{dynamics}",
        dynamics,
        "ser",
        metrics=["mutualinfo"],
        path=PATH_TO_DATA / "figure2" / f"exact-{dynamics}",
        num_procs=num_procs,
        seed=seed,
    )
    N = 5
    E = 5
    T = np.unique(np.logspace(0, 4, 100).astype("int"))
    if dynamics == "sis":
        coupling = np.array([0.5, 2, 4]) * config.dynamics.recovery_prob
        config.dynamics.set_value("auto_infection_prob", 1e-4)
    elif dynamics == "ising":
        coupling = [0.5, 1, 2]
    elif dynamics == "cowan":
        coupling = [1, 2, 4]
    config.dynamics.set_coupling(coupling)

    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.dynamics.set_value("num_steps", T)
    config.metrics.mutualinfo.set_value("num_samples", 1000)
    config.metrics.mutualinfo.set_value("method", "exact")
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
    for dynamics in ["ising", "sis", "cowan"]:
        config = get_config(dynamics, num_procs=40, time="1:00:00", mem=12)
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
