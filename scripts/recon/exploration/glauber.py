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
    num_procs=32,
    time="24:00:00",
    mem=12,
    seed=None,
):
    data_model = "glauber"
    likelihood_type = "uniform"
    if graph_prior == "degree_corrected_stochastic_block_model":
        graph_prior = "stochastic_block_model"
        likelihood_type = "degree_corrected"
    block_prior_type = "hyper"
    degree_prior_type = "hyper"

    if expname is None:
        expname = f"recon-{graph_prior}-{data_model}"
    config = ExperimentConfig.reconstruction(
        expname,
        data_model,
        graph_prior,
        metrics=["recon_information"],
        path=PATH_TO_DATA / "exploration" / f"recon-{graph_prior}-{data_model}",
        num_procs=num_procs,
        seed=seed,
    )
    N = 100
    E = 250
    T = 500
    coupling = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 4, 10)])
    config.data_model.set_value("coupling", coupling)
    config.data_model.set_value("num_active", 50)
    config.graph_prior.set_value("size", N)
    config.graph_prior.set_value("likelihood_type", likelihood_type)
    config.graph_prior.set_value("block_prior_type", block_prior_type)
    config.graph_prior.set_value("degree_prior_type", degree_prior_type)
    config.graph_prior.set_value("edge_count", E)
    config.data_model.set_value("num_steps", T)
    config.metrics.recon_information.set_value("num_samples", num_procs)
    config.metrics.recon_information.set_value("burn_per_vertex", 10)
    config.metrics.recon_information.set_value("start_from_original", False)
    config.metrics.recon_information.set_value("equilibrate_mode_cluster", True)
    config.metrics.recon_information.set_value("initial_burn", 2000)
    config.metrics.recon_information.set_value("num_sweeps", 1000)
    config.metrics.recon_information.set_value("method", "meanfield")

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
        expname = f"recon-{graph_prior}-glauber"
        config = get_config(
            expname, graph_prior=graph_prior, num_procs=40, time="24:00:00", mem=12
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
