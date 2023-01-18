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


def get_config(data_model="sis", num_procs=32, time="24:00:00", mem=12, seed=None):
    config = ExperimentConfig.reconstruction(
        f"figure2-exact-{data_model}",
        data_model,
        "erdosrenyi",
        metrics=["recon_information"],
        path=PATH_TO_DATA / "figure2" / f"exact-{data_model}",
        num_procs=num_procs,
        seed=seed,
    )
    N = 5
    E = 5
    T = np.unique(np.logspace(1, 4, 100).astype("int"))
    if data_model == "sis":
        config.data_model.set_value("recovery_prob", 0.1)
        config.data_model.set_value("infection_prob", [0.25, 0.5, 1])
        config.data_model.set_value("auto_infection_prob", 1e-4)
    elif data_model == "glauber":
        config.data_model.set_value("coupling", [0.25, 0.5, 1])
    elif data_model == "cowan":
        config.data_model.set_value("nu", [0.5, 1, 2])
        config.data_model.set_value("eta", 0.1)
    config.data_model.set_value("length", T)

    config.prior.set_value("size", N)
    config.prior.set_value("edge_count", E)
    config.prior.set_value("with_self_loops", False)
    config.prior.set_value("with_parallel_edges", False)
    config.metrics.recon_information.set_value("num_samples", 100 * num_procs)
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
    for data_model in ["glauber", "sis", "cowan"]:
        config = get_config(data_model, num_procs=40, time="24:00:00", mem=12)
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
