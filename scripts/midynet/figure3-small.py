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


def get_config_base(
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
    config.graph.set_value("size", N)
    config.graph.edge_count.set_value("state", E)
    config.metrics.mutualinfo.set_value("num_samples", 1000)
    config.metrics.mutualinfo.set_value(
        "method", ["annealed", "exact", "meanfield"]
    )

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


def get_config_vs_time(**kwargs):
    config = get_config_base(**kwargs)
    T = np.unique(np.logspace(0, 3, 20).astype("int"))
    dynamics = kwargs.get("dynamics")
    if dynamics == "sis":
        coupling = np.array([0.1, 0.5, 1]) / config.dynamics.recovery_prob
    elif dynamics == "ising":
        coupling = [0.5, 1, 2]
    elif dynamics == "cowan":
        coupling = [1, 1.5, 2]
    config.dynamics.set_coupling(coupling)
    config.dynamics.set_value("num_steps", T)
    return config


def get_config_vs_coupling(**kwargs):
    config = get_config_base(**kwargs)
    T = [10, 100, 500]
    dynamics = kwargs.get("dynamics")
    if dynamics == "sis":
        coupling = np.linspace(0, 1, 20) / config.dynamics.recovery_prob
    elif dynamics == "ising":
        coupling = np.linspace(0, 4, 20)
    elif dynamics == "cowan":
        coupling = np.linspace(0, 4, 20)
    config.dynamics.set_coupling(coupling)
    config.dynamics.set_value("num_steps", T)
    return config


def launch(config):
    script = ScriptManager(
        executable=PATH_TO_RUN_EXEC["run"],
        execution_command=EXECUTION_COMMAND,
        path_to_scripts="./scripts",
    )
    ais_config, exact_config, mf_config = script.split_param(
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

    exact_config.resources["time"] = "1:00:00"
    script.run(
        exact_config,
        resources=ais_config.resources,
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


def main():
    for dynamics in ["sis", "ising", "cowan"]:
        config = get_config_vs_time(dynamics, num_procs=40, mem=12)
        launch(config)

        config = get_config_vs_coupling(dynamics, num_procs=40, mem=12)
        launch(config)


if __name__ == "__main__":
    main()
