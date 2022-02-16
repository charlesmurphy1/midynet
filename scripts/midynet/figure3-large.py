import numpy as np
from load_specs import (
    PATH_TO_DATA,
    PATH_TO_LOG,
    PATH_TO_RUN_EXEC,
    EXECUTION_COMMAND,
    SPECS,
)
from midynet.config import ExperimentConfig
from midynet.scripts import ScriptManager


def get_config(
    dynamics="sis", num_procs=4, time="24:00:00", mem=12, seed=None
):
    config = ExperimentConfig.default(
        f"figure3-large-{dynamics}",
        dynamics,
        "ser",
        metrics=["mutualinfo"],
        path=PATH_TO_DATA / "figure3",
        seed=seed,
        num_procs=num_procs,
    )
    N, E = 100, 250
    T = 1000
    if dynamics == "sis":
        coupling = np.linspace(0, 1, 20)
        config.dynamics.set_coupling(coupling)
        config.dynamics.set_value("normalize", False)
    else:
        coupling = np.concatenate(
            [np.linspace(0, 1, 10), np.linspace(1, 4, 15)]
        )
        config.dynamics.set_coupling(coupling)
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.dynamics.set_value("num_steps", T)
    config.metrics.mutualinfo.set_value(
        "num_samples", max(1, 32 // num_procs) * num_procs
    )
    config.metrics.mutualinfo.set_value("method", ["meanfield", "annealed"])

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
        config = get_config(dynamics, num_procs=4, mem=12)
        script = ScriptManager(
            executable=PATH_TO_RUN_EXEC["run"],
            execution_command=EXECUTION_COMMAND,
            path_to_scripts="./scripts",
            path_to_log=PATH_TO_LOG,
        )
        ais_config, mf_config = script.split_param(
            config, "metrics.mutualinfo.method"
        )

        # mf_config.resources["time"] = "12:00:00"
        # mf_config.metrics.mutualinfo.set_value("num_sweeps", 1000)
        # script.run(
        #     mf_config,
        #     resources=mf_config.resources,
        #     modules_to_load=SPECS["modules_to_load"],
        #     virtualenv=SPECS["virtualenv"],
        #     extra_args=dict(verbose=2),
        #     teardown=False,
        # )

        ais_config.resources["time"] = "36:00:00"
        ais_config.metrics.mutualinfo.set_value("num_sweeps", 500)
        ais_config.metrics.mutualinfo.set_value("num_betas", 20)
        ais_config.metrics.mutualinfo.set_value("exp_betas", 1)
        script.run(
            ais_config,
            resources=ais_config.resources,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
            teardown=False,
        )


if __name__ == "__main__":
    main()
