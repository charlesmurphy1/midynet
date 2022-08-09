import pytest
from itertools import product

from midynet.config import (
    RandomGraphConfig,
    RandomGraphFactory,
    DynamicsConfig,
    DynamicsFactory,
    MetricsFactory,
    ExperimentConfig,
)

random_graph_setup = [
    # pytest.param(
    #     RandomGraphConfig.uniform_sbm(100, 250, 10),
    #     RandomGraphFactory,
    #     lambda obj: obj.sample(),
    #     id="sbm.uniform",
    # ),
    # pytest.param(
    #     RandomGraphConfig.hyperuniform_sbm(100, 250, 10),
    #     RandomGraphFactory,
    #     lambda obj: obj.sample(),
    #     id="sbm.hyperuniform",
    # ),
    # pytest.param(
    #     RandomGraphConfig.er(100, 250),
    #     RandomGraphFactory,
    #     lambda obj: obj.sample(),
    #     id="er.delta",
    # ),
    # pytest.param(
    #     RandomGraphConfig.ser(100, 250.0),
    #     RandomGraphFactory,
    #     lambda obj: obj.sample(),
    #     id="er.poisson",
    # ),
    # pytest.param(
    #     RandomGraphConfig.ser(100, 250),
    #     RandomGraphFactory,
    #     lambda obj: obj.sample(),
    #     id="ser",
    # ),
    # pytest.param(
    #     RandomGraphConfig.uniform_dcsbm(100, 250, 10),
    #     RandomGraphFactory,
    #     lambda obj: obj.sample(),
    #     id="dcsbm.uniform",
    # ),
    # pytest.param(
    #     RandomGraphConfig.uniform_cm(100, 250),
    #     RandomGraphFactory,
    #     lambda obj: obj.sample(),
    #     id="cm.uniform",
    # ),
    # pytest.param(
    #     RandomGraphConfig.poisson_cm(100, 250),
    #     RandomGraphFactory,
    #     lambda obj: obj.sample(),
    #     id="cm.poisson",
    # ),
    # pytest.param(
    #     RandomGraphConfig.nbinom_cm(100, 250, 0.2),
    #     RandomGraphFactory,
    #     lambda obj: obj.sample(),
    #     id="cm.nbinom",
    # ),
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
]


# def sample_dynamics(obj):
#     c = RandomGraphConfig.er(10, 25)
#     g = RandomGraphFactory.build(c)
#     obj.set_graph_prior(g)
#     obj.set_num_steps(10)
#     obj.sample()
#
#
# dynamics_setup = [
#     pytest.param(
#         DynamicsConfig.glauber(),
#         DynamicsFactory,
#         sample_dynamics,
#         id="glauber",
#     ),
#     pytest.param(
#         DynamicsConfig.sis(),
#         DynamicsFactory,
#         sample_dynamics,
#         id="sis",
#     ),
#     pytest.param(
#         DynamicsConfig.cowan(),
#         DynamicsFactory,
#         sample_dynamics,
#         id="cowan",
#     ),
#     pytest.param(
#         DynamicsConfig.degree(),
#         DynamicsFactory,
#         sample_dynamics,
#         id="degree",
#     ),
# ]

# metrics_setup = [
#     pytest.param(
#         ExperimentConfig.reconstruction(
#             "test", "glauber", "er", metrics=["dynamics_entropy"]
#         ),
#         MetricsFactory,
#         lambda obj: None,
#         id="metrics",
#     )
# ]


@pytest.mark.parametrize(
    "config, factory, run",
    [
        *random_graph_setup,
        # *dynamics_setup,
        # *metrics_setup,
    ],
)
def test_build_from_config(config, factory, run):
    factory.build(config)


@pytest.mark.parametrize(
    "config, factory, run",
    [
        *random_graph_setup,
        # *metrics_setup,
        # *dynamics_setup,
    ],
)
def test_run_after_creation(config, factory, run):
    obj = factory.build(config)
    run(obj)


# mcmc_setup = [
#     pytest.param(
#         ExperimentConfig.reconstruction("test", d, g),
#         MCMCFactory,
#         id=f"mcmc.{d}.{g}",
#     )
#     for d, g in product(
#         ["glauber", "sis", "cowan"],
#         [
#             "er",
#             "nbinom_cm",
#             "uniform_cm",
#             "hyperuniform_cm",
#             "uniform_sbm",
#             "hyperuniform_sbm",
#             "uniform_dcsbm",
#             "hyperuniform_dcsbm",
#         ],
#     )
# ]
#
#
# @pytest.mark.parametrize("config, factory", [*mcmc_setup])
# def test_build_reconstruction_from_exp_config(config, factory):
#     factory.build_reconstruction(config)
#
#
# @pytest.mark.parametrize("config, factory", [*mcmc_setup])
# def test_run_reconstruction_after_creation(config, factory):
#     obj = factory.build_reconstruction(config)
#     obj.others["dynamics"].sample()
#     obj.set_up()
#     obj.do_MH_sweep(1000)


if __name__ == "__main__":
    pass
