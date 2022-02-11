import numpy as np
from config import (
    get_config_figure2Exact,
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
    T = np.linspace(1, 100, 100).astype("int")
    if dynamics == "sis":
        coupling = np.linspace(0, 1, 10)
        config.dynamics.set_coupling(coupling)
        config.dynamics.set_value("normalize", False)
    else:
        coupling = np.linspace(0, 4, 10)
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
        "output": PATH_TO_LOG / f"{config.name}.out",
    }
    config.insert("resources", resources, force_non_sequence=True, unique=True)
    return config


def main():
    for dynamics in ["sis", "ising", "cowan"]:
        config = get_config(
            dynamics, num_procs=32, time="5:00:00", mem=12
        )
        script = ScriptManager(
            executable=PATH_TO_RUN_EXEC["run"],
            execution_command=EXECUTION_COMMAND,
            path_to_scripts="./scripts",
        )
        script.run(
            config,
            resources=config.resources,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
            teardown=False,
        )


if __name__ == "__main__":
    main()
