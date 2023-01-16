import numpy as np
import pathlib

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
        f"figure3-small-{dynamics}",
        dynamics,
        "ser",
        metrics=["mutualinfo"],
        path=PATH_TO_DATA / "figure3",
        seed=seed,
        num_procs=num_procs,
    )
    N, E = 5, 5
    T = 100
    if dynamics == "sis":
        coupling = np.linspace(0, 1, 20)
        config.dynamics.set_coupling(coupling)
        config.dynamics.set_value("normalize", False)
    else:
        coupling = np.concatenate(
            [np.linspace(0, 1, 10), np.linspace(1.1, 4, 15)]
        )
        config.dynamics.set_coupling(coupling)
    config.dynamics.set_value("num_steps", T)
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.metrics.mutualinfo.set_value("num_samples", 1000)
    config.metrics.mutualinfo.set_value(
        "method", ["annealed", "exact", "full-meanfield", "meanfield"]
    )

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
    for dynamics in ["ising"]:
        config = get_config(dynamics=dynamics, num_procs=40, mem=12)
        script = ScriptManager(
            executable=PATH_TO_RUN_EXEC["run"],
            execution_command=EXECUTION_COMMAND,
            path_to_scripts="./scripts",
            path_to_log=pathlib.Path(PATH_TO_LOG),
        )
        ais_config, exact_config, fmf_config, mf_config = script.split_param(
            config, "metrics.mutualinfo.method"
        )

        ais_config.resources["time"] = "16:00:00"
        script.run(
            ais_config,
            resources=ais_config.resources,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
            teardown=False,
        )

        exact_config.resources["time"] = "0:10:00"
        if dynamics == "sis":
            coupling = np.linspace(0, 1, 100)
            exact_config.dynamics.set_value("normalize", False)
        else:
            coupling = np.linspace(0, 4, 100)
        exact_config.dynamics.set_coupling(coupling)
        script.run(
            exact_config,
            resources=ais_config.resources,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
            teardown=False,
        )

        fmf_config.resources["time"] = "06:00:00"
        script.run(
            fmf_config,
            resources=fmf_config.resources,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
            teardown=False,
        )


        mf_config.resources["time"] = "04:00:00"
        script.run(
            mf_config,
            resources=mf_config.resources,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
            teardown=False,
        )


if __name__ == "__main__":
    main()
