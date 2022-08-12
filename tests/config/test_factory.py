import pytest
from itertools import product

from midynet.config import (
    RandomGraphConfig,
    RandomGraphFactory,
    DataModelConfig,
    DataModelFactory,
    MetricsFactory,
    ExperimentConfig,
    ReconstructionMCMC,
    PartitionMCMC,
)

random_graph_setup = [
    pytest.param(
        RandomGraphConfig.erdosrenyi(100, 250),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="er",
    ),
    pytest.param(
        RandomGraphConfig.configuration(100, 250),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="cm",
    ),
    # pytest.param(
    #     RandomGraphConfig.poisson(100, 250),
    #     RandomGraphFactory,
    #     lambda obj: obj.sample(),
    #     id="cm",
    # ),
    # pytest.param(
    #     RandomGraphConfig.nbinom(100, 250),
    #     RandomGraphFactory,
    #     lambda obj: obj.sample(),
    #     id="cm",
    # ),
    pytest.param(
        RandomGraphConfig.stochastic_block_model(100, 250),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="sbm",
    ),
    pytest.param(
        RandomGraphConfig.stochastic_block_model(
            100, 250, likelihood_type="degree_corrected"
        ),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="dcsbm",
    ),
    pytest.param(
        RandomGraphConfig.stochastic_block_model(100, 250),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="sbm",
    ),
    pytest.param(
        RandomGraphConfig.stochastic_block_model(
            100, 250, label_graph_prior_type="nested"
        ),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="hsbm",
    ),
    pytest.param(
        RandomGraphConfig.stochastic_block_model(
            100,
            250,
            likelihood_type="degree_corrected",
            label_graph_prior_type="nested",
        ),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="hdcsbm",
    ),
    pytest.param(
        RandomGraphConfig.planted_partition(sizes=[25, 25, 25, 25], edge_count=250),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="pp",
    ),
]


def sample_dynamics(obj):
    c = RandomGraphConfig.erdosrenyi(10, 25)
    g = RandomGraphFactory.build(c)
    obj.set_graph_prior(g)
    obj.set_num_steps(10)
    obj.sample()


data_setup = [
    pytest.param(DataModelConfig.sis(), DataModelFactory, sample_dynamics, id="sis"),
    pytest.param(
        DataModelConfig.glauber(), DataModelFactory, sample_dynamics, id="glauber"
    ),
    pytest.param(
        DataModelConfig.cowan(), DataModelFactory, sample_dynamics, id="cowan"
    ),
    pytest.param(
        DataModelConfig.degree(), DataModelFactory, sample_dynamics, id="degree"
    ),
]


metrics_setup = [
    pytest.param(
        ExperimentConfig.reconstruction(
            "test", "glauber", "erdosrenyi", metrics=["data_entropy"]
        ),
        MetricsFactory,
        lambda obj: None,
        id="metrics",
    )
]


@pytest.mark.parametrize(
    "config, factory, run",
    [
        *random_graph_setup,
        *data_setup,
        *metrics_setup,
    ],
)
def test_build_from_config(config, factory, run):
    factory.build(config)


@pytest.mark.parametrize(
    "config, factory, run",
    [
        *random_graph_setup,
        *metrics_setup,
        *data_setup,
    ],
)
def test_run_after_creation(config, factory, run):
    obj = factory.build(config)
    run(obj)


mcmc_setup = [
    pytest.param(
        ExperimentConfig.reconstruction("test", d, g),
        ReconstructionMCMC,
        id=f"mcmc.{d}.{g}",
    )
    for d, g in product(
        ["glauber", "sis", "cowan"],
        [
            "erdosrenyi",
            "configuration",
            "stochastic_block_model",
        ],
    )
]


@pytest.mark.parametrize("config, factory", [*mcmc_setup])
def test_run_reconstruction_from_exp_config(config, factory):
    graph = RandomGraphFactory.build(config.graph)
    dynamics = DataModelFactory.build(config.data_model)
    dynamics.set_graph_prior(graph)
    mcmc = ReconstructionMCMC(dynamics, graph)
    dynamics.sample()
    mcmc.do_MH_sweep(1000)


if __name__ == "__main__":
    pass
