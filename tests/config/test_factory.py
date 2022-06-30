import pytest
from itertools import product

from midynet.config import (
    EdgeCountPriorConfig,
    EdgeCountPriorFactory,
    BlockCountPriorConfig,
    BlockCountPriorFactory,
    BlockPriorConfig,
    BlockPriorFactory,
    EdgeMatrixPriorConfig,
    EdgeMatrixPriorFactory,
    DegreePriorConfig,
    DegreePriorFactory,
    RandomGraphConfig,
    RandomGraphFactory,
    DynamicsConfig,
    DynamicsFactory,
    MCMCFactory,
    MetricsFactory,
    ExperimentConfig,
)

edge_count_setup = [
    pytest.param(
        EdgeCountPriorConfig.delta(5),
        EdgeCountPriorFactory,
        lambda obj: obj.sample(),
        id="edge_count.delta",
    ),
    pytest.param(
        EdgeCountPriorConfig.poisson(5),
        EdgeCountPriorFactory,
        lambda obj: obj.sample(),
        id="edge_count.poisson",
    ),
]

block_count_setup = [
    pytest.param(
        BlockCountPriorConfig.delta(5),
        BlockCountPriorFactory,
        lambda obj: obj.sample(),
        id="block_count.delta",
    ),
    pytest.param(
        BlockCountPriorConfig.poisson(5),
        BlockCountPriorFactory,
        lambda obj: obj.sample(),
        id="block_count.poisson",
    ),
    pytest.param(
        BlockCountPriorConfig.uniform(5),
        BlockCountPriorFactory,
        lambda obj: obj.sample(),
        id="block_count.uniform",
    ),
]

block_setup = [
    pytest.param(
        BlockPriorConfig.delta([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        BlockPriorFactory,
        lambda obj: obj.sample(),
        id="block.delta",
    ),
    pytest.param(
        BlockPriorConfig.uniform(10, 1, 10),
        BlockPriorFactory,
        lambda obj: obj.sample(),
        id="block.uniform",
    ),
    pytest.param(
        BlockPriorConfig.hyperuniform(10, 1, 10),
        BlockPriorFactory,
        lambda obj: obj.sample(),
        id="block.uniform",
    ),
]


def sample_edge_matrix(obj):
    c = BlockPriorConfig.uniform(10, 1, 10)
    c.block_count.max = 10
    b = BlockPriorFactory.build(BlockPriorConfig.uniform(10, 1, 10))
    obj.set_block_prior(b.wrap)
    obj.sample()


edge_matrix_setup = [
    pytest.param(
        EdgeMatrixPriorConfig.uniform(10),
        EdgeMatrixPriorFactory,
        sample_edge_matrix,
        id="edge_matrix.uniform",
    ),
]


def sample_degrees(obj):
    b = BlockPriorFactory.build(BlockPriorConfig.uniform(10, 1, 10))
    b.set_size(100)

    e = EdgeMatrixPriorFactory.build(EdgeMatrixPriorConfig.uniform(250))
    e.set_block_prior(b.wrap)

    obj.set_block_prior(b.wrap)
    obj.set_edge_matrix_prior(e.wrap)
    obj.sample()


degrees_setup = [
    pytest.param(
        DegreePriorConfig.uniform(),
        DegreePriorFactory,
        sample_degrees,
        id="degree.uniform",
    ),
]

random_graph_setup = [
    pytest.param(
        RandomGraphConfig.uniform_sbm(100, 250, 10),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="sbm.uniform",
    ),
    pytest.param(
        RandomGraphConfig.hyperuniform_sbm(100, 250, 10),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="sbm.hyperuniform",
    ),
    pytest.param(
        RandomGraphConfig.er(100, 250),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="er.delta",
    ),
    pytest.param(
        RandomGraphConfig.ser(100, 250.0),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="er.poisson",
    ),
    pytest.param(
        RandomGraphConfig.ser(100, 250),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="ser",
    ),
    pytest.param(
        RandomGraphConfig.uniform_dcsbm(100, 250, 10),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="dcsbm.uniform",
    ),
    pytest.param(
        RandomGraphConfig.uniform_cm(100, 250),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="cm.uniform",
    ),
    pytest.param(
        RandomGraphConfig.poisson_cm(100, 250),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="cm.poisson",
    ),
    pytest.param(
        RandomGraphConfig.nbinom_cm(100, 250, 0.2),
        RandomGraphFactory,
        lambda obj: obj.sample(),
        id="cm.nbinom",
    ),
]


def sample_dynamics(obj):
    c = RandomGraphConfig.er(10, 25)
    g = RandomGraphFactory.build(c)
    obj.set_graph_prior(g.wrap)
    obj.set_num_steps(10)
    obj.sample()


dynamics_setup = [
    pytest.param(
        DynamicsConfig.glauber(),
        DynamicsFactory,
        sample_dynamics,
        id="glauber",
    ),
    pytest.param(
        DynamicsConfig.sis(),
        DynamicsFactory,
        sample_dynamics,
        id="sis",
    ),
    pytest.param(
        DynamicsConfig.cowan(),
        DynamicsFactory,
        sample_dynamics,
        id="cowan",
    ),
    pytest.param(
        DynamicsConfig.degree(),
        DynamicsFactory,
        sample_dynamics,
        id="degree",
    ),
]

metrics_setup = [
    pytest.param(
        ExperimentConfig.reconstruction(
            "test", "glauber", "er", metrics=["dynamics_entropy"]
        ),
        MetricsFactory,
        lambda obj: None,
        id="metrics",
    )
]


@pytest.mark.parametrize(
    "config, factory, run",
    [
        *edge_count_setup,
        *block_count_setup,
        *block_setup,
        *edge_matrix_setup,
        *degrees_setup,
        *random_graph_setup,
        *dynamics_setup,
        *metrics_setup,
    ],
)
def test_build_from_config(config, factory, run):
    factory.build(config)


@pytest.mark.parametrize(
    "config, factory, run",
    [
        *edge_count_setup,
        *block_count_setup,
        *block_setup,
        *edge_matrix_setup,
        *degrees_setup,
        *random_graph_setup,
        *dynamics_setup,
        *metrics_setup,
    ],
)
def test_run_after_creation(config, factory, run):
    obj = factory.build(config)
    run(obj)


mcmc_setup = [
    pytest.param(
        ExperimentConfig.reconstruction("test", d, g),
        MCMCFactory,
        id=f"mcmc.{d}.{g}",
    )
    for d, g in product(
        ["glauber", "sis", "cowan"],
        [
            "er",
            "nbinom_cm",
            "uniform_cm",
            "hyperuniform_cm",
            "uniform_sbm",
            "hyperuniform_sbm",
            "uniform_dcsbm",
            "hyperuniform_dcsbm",
        ],
    )
]


@pytest.mark.parametrize("config, factory", [*mcmc_setup])
def test_build_reconstruction_from_exp_config(config, factory):
    factory.build_reconstruction(config)


@pytest.mark.parametrize("config, factory", [*mcmc_setup])
def test_run_reconstruction_after_creation(config, factory):
    obj = factory.build_reconstruction(config)
    obj.others["dynamics"].sample()
    obj.set_up()
    obj.do_MH_sweep(1000)


if __name__ == "__main__":
    pass
