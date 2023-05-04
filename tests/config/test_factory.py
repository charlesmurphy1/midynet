import pytest
from itertools import product

from midynet.config import (
    GraphConfig,
    GraphFactory,
    DataModelConfig,
    DataModelFactory,
)

random_graph_setup = [
    pytest.param(
        GraphConfig.erdosrenyi(100, 250),
        GraphFactory,
        lambda obj: obj.sample(),
        id="er",
    ),
    pytest.param(
        GraphConfig.configuration(100, 250),
        GraphFactory,
        lambda obj: obj.sample(),
        id="cm",
    ),
    pytest.param(
        GraphConfig.poisson(100, 250),
        GraphFactory,
        lambda obj: obj.sample(),
        id="pcm",
    ),
    pytest.param(
        GraphConfig.nbinom(100, 250),
        GraphFactory,
        lambda obj: obj.sample(),
        id="nbcm",
    ),
    pytest.param(
        GraphConfig.stochastic_block_model(100, 250),
        GraphFactory,
        lambda obj: obj.sample(),
        id="sbm",
    ),
    pytest.param(
        GraphConfig.stochastic_block_model(
            100, 250, likelihood_type="degree_corrected"
        ),
        GraphFactory,
        lambda obj: obj.sample(),
        id="dcsbm",
    ),
    pytest.param(
        GraphConfig.stochastic_block_model(100, 250),
        GraphFactory,
        lambda obj: obj.sample(),
        id="sbm",
    ),
    pytest.param(
        GraphConfig.stochastic_block_model(100, 250, label_graph_prior_type="nested"),
        GraphFactory,
        lambda obj: obj.sample(),
        id="hsbm",
    ),
    pytest.param(
        GraphConfig.stochastic_block_model(
            100,
            250,
            likelihood_type="degree_corrected",
            label_graph_prior_type="nested",
        ),
        GraphFactory,
        lambda obj: obj.sample(),
        id="hdcsbm",
    ),
]


def sample_dynamics(obj):
    c = GraphConfig.erdosrenyi(10, 25)
    g = GraphFactory.build(c)
    obj.set_graph_prior(g)
    obj.set_length(10)
    obj.sample()


data_setup = [
    pytest.param(DataModelConfig.sis(), DataModelFactory, sample_dynamics, id="sis"),
    pytest.param(
        DataModelConfig.glauber(),
        DataModelFactory,
        sample_dynamics,
        id="glauber",
    ),
    pytest.param(
        DataModelConfig.cowan(), DataModelFactory, sample_dynamics, id="cowan"
    ),
    pytest.param(
        DataModelConfig.degree(),
        DataModelFactory,
        sample_dynamics,
        id="degree",
    ),
]


# metrics_setup = [
#     pytest.param(
#         ExperimentConfig.reconstruction(
#             "test", "glauber", "erdosrenyi", metrics=["recon_information"]
#         ),
#         MetricsFactory,
#         lambda obj: None,
#         id="metrics",
#     )
# ]


@pytest.mark.parametrize(
    "config, factory, run",
    [*random_graph_setup, *data_setup],
)
def test_build_from_config(config, factory, run):
    run(factory.build(config))


@pytest.mark.parametrize(
    "config, factory, run",
    [
        *random_graph_setup,
        *data_setup,
    ],
)
def test_run_after_creation(config, factory, run):
    obj = factory.build(config)
    run(obj)


# def test_loading_graph():
#     c = GraphConfig.auto("gt-karate")
#     g = GraphFactory.build(c)
#     assert g.get_size() == 34
#     assert g.get_total_edge_number() == 78


if __name__ == "__main__":
    pass
