import pytest

from midynet.config import *

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
        BlockPriorConfig.uniform(),
        BlockPriorFactory,
        lambda obj: obj.sample(),
        id="block.uniform",
    ),
    pytest.param(
        BlockPriorConfig.hyperuniform(),
        BlockPriorFactory,
        lambda obj: obj.sample(),
        id="block.uniform",
    ),
]


def sample_edge_matrix(obj):
    c = BlockPriorConfig.uniform()
    c.block_count.max = 10
    b = BlockPriorFactory.build(BlockPriorConfig.uniform())
    obj.set_block_prior(b.get_wrap())
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
    b = BlockPriorFactory.build(BlockPriorConfig.uniform())
    b.set_size(100)

    e = EdgeMatrixPriorFactory.build(EdgeMatrixPriorConfig.uniform(250))
    e.set_block_prior(b.get_wrap())

    obj.set_block_prior(b.get_wrap())
    obj.set_edge_matrix_prior(e.get_wrap())
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
    obj.set_random_graph(g.get_wrap())
    obj.set_num_steps(10)
    obj.sample()


dynamics_setup = {
    pytest.param(
        DynamicsConfig.ising(),
        DynamicsFactory,
        sample_dynamics,
        id="ising",
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
}

random_graph_mcmc_setup = [
    pytest.param(
        RandomGraphConfig.uniform_sbm(100, 250, 10),
        RandomGraphMCMCFactory,
        lambda obj: None,
        id="mcmc.sbm.uniform",
    ),
    pytest.param(
        RandomGraphConfig.hyperuniform_sbm(100, 250, 10),
        RandomGraphMCMCFactory,
        lambda obj: None,
        id="mcmc.sbm.hyperuniform",
    ),
    pytest.param(
        RandomGraphConfig.er(100, 250),
        RandomGraphMCMCFactory,
        lambda obj: None,
        id="mcmc.er.delta",
    ),
    pytest.param(
        RandomGraphConfig.ser(100, 250.0),
        RandomGraphMCMCFactory,
        lambda obj: None,
        id="mcmc.er.poisson",
    ),
    pytest.param(
        RandomGraphConfig.ser(100, 250),
        RandomGraphMCMCFactory,
        lambda obj: None,
        id="mcmc.ser",
    ),
    pytest.param(
        RandomGraphConfig.uniform_dcsbm(100, 250, 10),
        RandomGraphMCMCFactory,
        lambda obj: None,
        id="mcmc.dcsbm.uniform",
    ),
    pytest.param(
        RandomGraphConfig.uniform_cm(100, 250),
        RandomGraphMCMCFactory,
        lambda obj: None,
        id="mcmc.cm.uniform",
    ),
    pytest.param(
        RandomGraphConfig.poisson_cm(100, 250),
        RandomGraphMCMCFactory,
        lambda obj: None,
        id="mcmc.cm.poisson",
    ),
    pytest.param(
        RandomGraphConfig.nbinom_cm(100, 250, 0.2),
        RandomGraphMCMCFactory,
        lambda obj: None,
        id="mcmc.cm.nbinom",
    ),
]

metrics_setup = [
    pytest.param(
        ExperimentConfig.default("test", "ising", "er", metrics=["dynamics_entropy"]),
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
        *random_graph_mcmc_setup,
        *metrics_setup,
    ],
)
def test_build_from_config(config, factory, run):
    obj = factory.build(config)


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
        *random_graph_mcmc_setup,
        *metrics_setup,
    ],
)
def test_run_after_creation(config, factory, run):
    obj = factory.build(config)
    run(obj)


if __name__ == "__main__":
    pass
