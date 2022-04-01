import time

from _midynet.mcmc import DynamicsMCMC
from _midynet.mcmc.callbacks import CollectEdgeMultiplicityOnSweep
from midynet.config import (
    DynamicsFactory,
    RandomGraphFactory,
    RandomGraphMCMCFactory,
    MCMCVerboseFactory,
    ExperimentConfig,
    Wrapper,
)


def makeConfig():
    c = ExperimentConfig.default("test", "sis", "nbinom_cm")
    c.graph.set_value("size", 1000)
    c.graph.edge_count.set_value("state", 2500)
    c.graph.set_value("sample_graph_prior_prob", 0.0)
    c.graph.set_value("heterogeneity", 0.0)
    c.dynamics.set_value("num_steps", 2000)
    c.dynamics.set_coupling(0.3)
    c.dynamics.set_value("recovery_prob", 0.5)
    c.dynamics.set_value("auto_infection_prob", 0.001)
    c.dynamics.set_value("initial_active", 1)
    c.insert("burn_per_vertex", 5)
    c.insert("num_betas", 10)
    c.insert("exp_betas", 0.5)
    c.insert("start_from_original", True)
    c.insert("initial_burn", 10000)
    c.insert("num_sweeps", 100)
    return c


def makeMCMC(cfg):
    randomGraph = RandomGraphFactory.build(cfg.graph)
    dynamics = DynamicsFactory.build(cfg.dynamics)
    dynamics.set_random_graph(randomGraph.get_wrap())
    randomGraphMCMC = RandomGraphMCMCFactory.build(cfg.graph)
    mcmc = DynamicsMCMC(dynamics, randomGraphMCMC.get_wrap(), 1, 1, 0)
    return Wrapper(
        mcmc,
        randomGraph=randomGraph,
        dynamics=dynamics,
        randomGraphMCMC=randomGraphMCMC,
    )


def main():
    cfg = makeConfig()
    print(cfg.format())
    mcmc = makeMCMC(cfg)
    callback = CollectEdgeMultiplicityOnSweep()
    mcmc.add_callback(callback)
    verbose = MCMCVerboseFactory.build("console")
    mcmc.add_callback(verbose.get_wrap())
    dynamics = mcmc.get_other("dynamics")
    dynamics.sample()
    mcmc.set_up()
    times = []
    for i in range(200):
        t0 = time.time()
        s, f = mcmc.do_MH_sweep(cfg.burn_per_vertex * cfg.graph.size)
        t1 = time.time()
        times.append(t1 - t0)
    print(times)

if __name__ == "__main__":
    main()
