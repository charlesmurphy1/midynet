import importlib

import numpy as np
from _midynet.mcmc import DynamicsMCMC
from _midynet.mcmc.callbacks import (
    CollectEdgeMultiplicityOnSweep,
    CollectLikelihoodOnSweep,
    CollectPartitionOnSweep,
)
from midynet.config import Config
from midynet.util import enumerate_all_graphs, log_mean_exp, log_sum_exp

__all__ = ("get_log_evidence", "get_log_posterior")


def get_log_evidence_arithmetic(dynamicsMCMC: DynamicsMCMC, config: Config):
    logp = []
    g = dynamicsMCMC.get_graph()
    for k in range(config.K):
        logp_k = []
        for m in range(config.num_sweeps):
            dynamicsMCMC.get_dynamics().sample_graph()
            logp_k.append(dynamicsMCMC.get_log_likelihood())
        logp.append(log_mean_exp(logp_k))
    dynamicsMCMC.set_graph(g)

    return np.mean(logp)


def get_log_evidence_harmonic(dynamicsMCMC: DynamicsMCMC, config: Config):
    callback = CollectLikelihoodOnSweep()
    g = dynamicsMCMC.get_graph()
    dynamicsMCMC.add_callback(callback)
    dynamicsMCMC.set_up()
    burn = config.burn_per_vertex * dynamicsMCMC.get_dynamics().get_size()
    s, f = dynamicsMCMC.do_MH_sweep(burn=config.initial_burn)
    for i in range(config.num_sweeps):
        s, f = dynamicsMCMC.do_MH_sweep(burn=burn)

    logp = -np.array(callback.get_log_likelihoods())

    dynamicsMCMC.tear_down()
    dynamicsMCMC.pop_callback()
    dynamicsMCMC.set_graph(g)
    return log_mean_exp(logp)


def get_log_evidence_meanfield(dynamicsMCMC: DynamicsMCMC, config: Config):
    return dynamicsMCMC.get_log_joint() - get_log_posterior_meanfield(
        dynamicsMCMC, config
    )


def get_log_evidence_annealed(
    dynamicsMCMC: DynamicsMCMC, config: Config, verbose=0
):
    callback = CollectLikelihoodOnSweep()
    g = dynamicsMCMC.get_graph()
    dynamicsMCMC.add_callback(callback)
    dynamicsMCMC.set_up()
    burn = config.burn_per_vertex * dynamicsMCMC.get_dynamics().get_size()
    logp = []

    beta_k = np.linspace(0, 1, config.num_betas) ** config.exp_betas
    for lb, ub in zip(beta_k[:-1], beta_k[1:]):
        if verbose:
            print(f"beta: {lb}")
        dynamicsMCMC.set_beta_likelihood(lb)
        if config.reset_to_original:
            dynamicsMCMC.set_graph(g)
        s, f = dynamicsMCMC.do_MH_sweep(burn=config.initial_burn)
        for i in range(config.num_sweeps):
            dynamicsMCMC.do_MH_sweep(burn=burn)
        logp_k = (ub - lb) * np.array(callback.get_log_likelihoods())
        logp.append(log_mean_exp(logp_k))
        callback.clear()
    dynamicsMCMC.tear_down()
    dynamicsMCMC.pop_callback()
    dynamicsMCMC.set_graph(g)

    return sum(logp)


def get_log_evidence_exact(dynamicsMCMC: DynamicsMCMC, config: Config):
    logevidence = []
    original_graph = dynamicsMCMC.get_graph()
    size = dynamicsMCMC.get_dynamics().get_size()
    edge_proposer = dynamicsMCMC.get_random_graph_mcmc().get_edge_proposer()
    graph = dynamicsMCMC.get_random_graph_mcmc().get_random_graph()
    edge_count = graph.get_edge_count()
    allow_self_loops = edge_proposer.allow_self_loops()
    allow_multiedges = edge_proposer.allow_multiedges()

    dynamicsMCMC.set_up()
    counter = 0
    for g in enumerate_all_graphs(
        size, edge_count, allow_self_loops, allow_multiedges
    ):
        counter += 1
        if graph.is_compatible(g):
            dynamicsMCMC.set_graph(g)
            logevidence.append(dynamicsMCMC.get_log_joint())
    dynamicsMCMC.tear_down()

    dynamicsMCMC.set_graph(original_graph)
    return log_sum_exp(logevidence)


def get_log_evidence_exact_meanfield(
    dynamicsMCMC: DynamicsMCMC, config: Config
):
    return dynamicsMCMC.get_log_joint() - get_log_posterior_exact_meanfield(
        dynamicsMCMC, config
    )


def get_log_evidence(dynamicsMCMC: DynamicsMCMC, config: Config):
    method = config.get_value("method", "meanfield")
    functions = {
        "exact": get_log_evidence_exact,
        "exact_meanfield": get_log_evidence_exact_meanfield,
        "arithmetic": get_log_evidence_arithmetic,
        "harmonic": get_log_evidence_harmonic,
        "meanfield": get_log_evidence_meanfield,
        "annealed": get_log_evidence_annealed,
    }
    if method in functions:
        return functions[method](dynamicsMCMC, config)
    else:
        message = (
            f"Invalid method {method}, valid methods"
            + f"are {list(functions.keys())}."
        )
        raise ValueError(message)


def get_log_posterior_arithmetic(dynamicsMCMC: DynamicsMCMC, config: Config):
    return dynamicsMCMC.get_log_joint() - get_log_evidence_arithmetic(
        dynamicsMCMC, config
    )


def get_log_posterior_harmonic(dynamicsMCMC: DynamicsMCMC, config: Config):
    return dynamicsMCMC.get_log_joint() - get_log_evidence_harmonic(
        dynamicsMCMC, config
    )


def get_log_posterior_meanfield(dynamicsMCMC: DynamicsMCMC, config: Config):
    graph_callback = CollectEdgeMultiplicityOnSweep()
    # verbose = MCMCVerboseFactory.build_console()
    dynamicsMCMC.add_callback(graph_callback)
    # dynamicsMCMC.add_callback(verbose.get_wrap())
    dynamicsMCMC.set_up()
    burn = config.burn_per_vertex * dynamicsMCMC.get_dynamics().get_size()
    s, f = dynamicsMCMC.do_MH_sweep(burn=config.initial_burn)

    for i in range(config.num_sweeps):
        _s, _f = dynamicsMCMC.do_MH_sweep(burn=burn)
        s += _s
        f += _f
    logp = -graph_callback.get_marginal_entropy()  # -H(G|X)

    dynamicsMCMC.tear_down()
    dynamicsMCMC.pop_callback()

    return logp


def get_log_posterior_meanfield_sbm(
    dynamicsMCMC: DynamicsMCMC, config: Config
):
    if importlib.util.find_spec("graph_tool") is None:
        message = (
            "The meanfield method cannot be used for SBM graphs, "
            + "because `graph_tool` is not installed."
        )
        raise NotImplementedError(message)
    else:
        from graph_tool.inference import (
            ModeClusterState,
            mcmc_equilibrate,
        )

        graph_callback = CollectEdgeMultiplicityOnSweep()
        partition_callback = CollectPartitionOnSweep()
        dynamicsMCMC.add_callback(graph_callback)
        dynamicsMCMC.add_callback(partition_callback)
        dynamicsMCMC.set_up()
        burn = config.burn_per_vertex * dynamicsMCMC.get_dynamics().get_size()
        for i in range(config.num_sweeps):
            dynamicsMCMC.do_MH_sweep(burn=burn)
        logp = -graph_callback.get_marginal_entropy()  # -H(G|X)
        partitions = partition_callback.get_partitions()  # -H(b|X)
        partition_modes = ModeClusterState(partitions)
        mcmc_equilibrate(
            partition_modes, wait=1, mcmc_args=dict(niter=1, beta=np.inf)
        )
        logp += -partition_modes.posterior_entropy(MLE=True)

        dynamicsMCMC.tear_down()
        dynamicsMCMC.pop_callback()
        dynamicsMCMC.pop_callback()

        return logp


def get_log_posterior_annealed(dynamicsMCMC: DynamicsMCMC, config: Config):
    return dynamicsMCMC.get_log_joint() - get_log_evidence_annealed(
        dynamicsMCMC, config
    )


def get_log_posterior_exact(dynamicsMCMC: DynamicsMCMC, config: Config):
    return dynamicsMCMC.get_log_joint() - get_log_evidence_exact(
        dynamicsMCMC, config
    )


def get_log_posterior_exact_meanfield(
    dynamicsMCMC: DynamicsMCMC, config: Config
):
    original_graph = dynamicsMCMC.get_graph()
    size = dynamicsMCMC.get_dynamics().get_size()
    edge_proposer = dynamicsMCMC.get_random_graph_mcmc().get_edge_proposer()
    graph = dynamicsMCMC.get_random_graph_mcmc().get_random_graph()
    edge_count = graph.get_edge_count()
    allow_self_loops = edge_proposer.allow_self_loops()
    allow_multiedges = edge_proposer.allow_multiedges()

    graph_callback = CollectEdgeMultiplicityOnSweep()
    dynamicsMCMC.add_callback(graph_callback)
    dynamicsMCMC.set_up()
    for g in enumerate_all_graphs(
        size, edge_count, allow_self_loops, allow_multiedges
    ):
        if graph.is_compatible(g):
            dynamicsMCMC.set_graph(g)
            graph_callback.collect()
    logp = -graph_callback.get_marginal_entropy()  # -H(G|X)

    dynamicsMCMC.tear_down()
    dynamicsMCMC.pop_callback()
    dynamicsMCMC.set_graph(original_graph)

    return logp


def get_log_posterior(dynamicsMCMC: DynamicsMCMC, config: Config):
    method = config.get_value("method", "meanfield")
    functions = {
        "exact": get_log_posterior_exact,
        "exact_meanfield": get_log_posterior_exact_meanfield,
        "arithmetic": get_log_posterior_arithmetic,
        "harmonic": get_log_posterior_harmonic,
        "meanfield": get_log_posterior_meanfield,
        "annealed": get_log_posterior_annealed,
    }
    if method in functions:
        return functions[method](dynamicsMCMC, config)
    else:
        message = (
            f"Invalid method {method}, valid methods"
            + f"are {list(functions.keys())}."
        )
        raise ValueError(message)
