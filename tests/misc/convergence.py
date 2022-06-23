import pytest

from netrd.distance import Frobenius
from midynet.config import (
    DynamicsFactory,
    RandomGraphFactory,
    MCMCFactory,
    ExperimentConfig,
    Wrapper,
)
from midynet.util import MCMCConvergenceAnalysis


@pytest.fixture
def config():
    c = ExperimentConfig.default("test", "ising", "nbinom_cm")
    c.graph.set_value("size", 4)
    c.graph.edge_count.set_value("state", 4)
    c.graph.set_value("sample_graph_prior_prob", 0.0)
    c.graph.set_value("heterogeneity", 0.0)
    c.dynamics.set_value("num_steps", [1])
    c.dynamics.set_coupling(0.0)
    # c.dynamics.set_value("infection_prob", 0.5)
    # c.dynamics.set_value("recovery_prob", 0.5)
    c.insert("num_sweeps", 10000)
    c.insert("numsteps_between_samples", 5)
    return c


# def mcmc_analysis(c, callbacks=None):
#     mcmc = MCMCFactory.build_reconstruction(c)
#     mcmc.others["dynamics"].sample()
#     mcmc.set_up()
#     measure = Frobenius().dist
#     return Wrapper(
#         MCMCConvergenceAnalysis(mcmc, measure, callbacks=callbacks),
#         D_MCMC=mcmc,
#         distance=measure,
#         g_mcmc=g_mcmc,
#         d=d,
#         g=g,
#     )


if __name__ == "__main__":
    pass
