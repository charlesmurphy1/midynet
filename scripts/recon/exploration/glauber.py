import numpy as np
import itertools
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
    expname=None,
    graph_prior="erdosrenyi",
    data_model="glauber",
    size=100,
    edge_count=250,
    length=1000,
    method="meanfield",
    num_procs=32,
    time="24:00:00",
    mem=12,
    seed=None,
):
    block_prior_type = "hyper"
    degree_prior_type = "hyper"

    samples_per_proc = 1 if size > 6 else 100

    expname = f"recon-{graph_prior}-{data_model}-{method}"
    if size < 6:
        expname = "small-" + expname
    path_to_exp = PATH_TO_DATA / "exploration" / expname

    if expname is None:
        expname = f"recon-{graph_prior}-{data_model}"
    config = ExperimentConfig.reconstruction(
        expname,
        data_model,
        graph_prior,
        metrics=["recon_information"],
        path=path_to_exp,
        num_procs=num_procs,
        seed=seed,
    )
    coupling = np.concatenate([np.linspace(0, 2, 20)])
    config.data_model.set_value("coupling", coupling)
    config.data_model.set_value("num_active", int(size / 2))
    config.graph_prior.set_value("size", size)
    config.graph_prior.set_value("block_prior_type", block_prior_type)
    config.graph_prior.set_value("degree_prior_type", degree_prior_type)
    config.graph_prior.set_value("edge_count", edge_count)
    config.data_model.set_value("num_steps", length)
    config.metrics.recon_information.set_value(
        "num_samples", samples_per_proc * num_procs
    )
    config.metrics.recon_information.set_value("burn_per_vertex", 10)
    config.metrics.recon_information.set_value("start_from_original", False)
    config.metrics.recon_information.set_value("equilibrate_mode_cluster", True)
    config.metrics.recon_information.set_value("initial_burn", 2000)
    config.metrics.recon_information.set_value("num_sweeps", 500)
    config.metrics.recon_information.set_value("method", method)

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
    for graph_prior in [
        "erdosrenyi",
        "configuration",
        "stochastic_block_model",
        "degree_corrected_stochastic_block_model",
    ]:
        config = get_config(
            data_model="glauber",
            graph_prior=graph_prior,
            size=4,
            edge_count=4,
            length=100,
            method="exact",
            num_procs=1,
            time="12:00:00",
            mem=12,
        )
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
