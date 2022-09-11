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


def get_config(dynamics="sis", num_procs=32, time="24:00:00", mem=12, seed=None, t=0):
    config = ExperimentConfig.reconstruction(
        f"test1-{dynamics}-t{t}",
        dynamics,
        "erdosrenyi",
        metrics=["recon_information"],
        path=PATH_TO_DATA / "exploration" / f"test1-{dynamics}-t{t}",
        num_procs=num_procs,
        seed=seed,
    )
    N = 5
    E = 5
    T = 20
    if dynamics == "sis":
        config.data_model.set_value("infection_prob", 0.5)
        config.data_model.set_value("recovery_prob", 0.5)
        config.data_model.set_value("auto_infection_prob", 1e-4)
    elif dynamics == "glauber":
        config.data_model.set_value("coupling", np.linspace(0, 2, 20))

    elif dynamics == "cowan":
        config.data_model.set_value("nu", 1)

    config.graph_prior.set_value("with_self_loops", False)
    config.graph_prior.set_value("with_parallel_edges", False)
    config.graph_prior.set_value("size", N)
    config.graph_prior.set_value("edge_count", E)
    config.data_model.set_value("past_length", t)
    config.data_model.set_value("length", T)
    config.data_model.set_value("num_active", -1)
    config.metrics.recon_information.set_value("num_samples", 500 * num_procs)
    config.metrics.recon_information.set_value("method", "exact")
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
    for t in [0, 5, 10, 15, 19]:
        config = get_config("glauber", num_procs=4, time="1:00:00", mem=12, t=t)
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
