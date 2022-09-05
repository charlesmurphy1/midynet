from __future__ import annotations
from collections import defaultdict
from scipy.special import loggamma
import numpy as np
import importlib
import time

from graphinf.random_graph import (
    BlockLabeledRandomGraph,
    NestedBlockLabeledRandomGraph,
)
from graphinf.mcmc import GraphReconstructionMCMC, PartitionReconstructionMCMC
from graphinf.mcmc.callbacks import (
    CollectEdgesOnSweep,
    CollectPartitionOnSweep,
    CollectLikelihoodOnSweep,
)
from graphinf.utility import get_weighted_edge_list

from midynet.config import Config
from midynet.utility import (
    enumerate_all_graphs,
    enumerate_all_partitions,
    log_mean_exp,
    log_sum_exp,
)

__all__ = (
    "get_log_evidence_arithmetic",
    "get_log_evidence_harmonic",
    "get_log_evidence_annealed",
    "get_log_evidence_exact",
    "get_log_posterior_meanfield",
    "get_log_posterior_exact_meanfield",
    "get_graph_log_evidence_meanfield",
    "get_graph_log_evidence_annealed",
    "get_graph_log_evidence_exact",
)


def get_log_evidence_arithmetic(data_model: DynamicsWrapper, config: Config):
    mcmc = GraphReconstructionMCMC(data_model)
    logp = []
    g = mcmc.get_graph()
    for k in range(config.K):
        logp_k = []
        for m in range(config.num_sweeps):
            mcmc.sample_prior()
            logp_k.append(mcmc.get_log_likelihood())
        logp.append(log_mean_exp(logp_k))
    mcmc.set_graph(g)

    return np.mean(logp)


def get_log_evidence_harmonic(
    data_model: DynamicsWrapper, config: Config, verbose: int = 0
):
    mcmc = GraphReconstructionMCMC(data_model, verbose=verbose)
    callback = CollectLikelihoodOnSweep()
    mcmc.insert_callback("collector", callback)

    g = mcmc.get_graph()
    burn = config.burn_per_vertex * data_model.get_size()
    s, f = mcmc.do_MH_sweep(burn=config.initial_burn)
    for i in range(config.num_sweeps):
        s, f = mcmc.do_MH_sweep(burn=burn)

    logp = -np.array(callback.get_data())

    mcmc.remove_callback("collector")
    if verbose:
        mcmc.remove_callback("verbose")
    mcmc.set_graph(g)
    return log_mean_exp(logp)


def get_log_evidence_annealed(
    data_model: DynamicsWrapper, config: Config, verbose: int = 0
):
    mcmc = GraphReconstructionMCMC(data_model, verbose=verbose)
    callback = CollectLikelihoodOnSweep()
    mcmc.insert_callback("collector", callback)
    original_graph = mcmc.get_graph()

    burn = config.burn_per_vertex * data_model.get_size()
    logp = []

    beta_k = np.linspace(0, 1, config.num_betas + 1) ** (1 / config.exp_betas)
    for lb, ub in zip(beta_k[:-1], beta_k[1:]):
        if verbose:
            print(f"beta: {lb}")
        mcmc.set_beta_likelihood(lb)
        if config.get_value("start_from_original", False):
            mcmc.set_graph(original_graph)
        else:
            mcmc.sample_prior()
        s, f = mcmc.do_MH_sweep(burn=config.initial_burn)
        for i in range(config.num_sweeps):
            mcmc.do_MH_sweep(burn=burn)
        logp_k = (ub - lb) * np.array(callback.get_data())
        logp.append(log_mean_exp(logp_k))
        callback.clear()

    mcmc.remove_callback("collector")
    if verbose:
        mcmc.remove_callback("verbose")
    mcmc.set_graph(original_graph)

    return sum(logp)


def get_log_evidence_exact(data_model: DynamicsWrapper, config: Config):
    logevidence = []
    original_graph = data_model.get_graph()
    size = data_model.get_size()
    edge_count = data_model.get_graph().get_total_edge_number()
    allow_self_loops = data_model.get_graph_prior().with_self_loops()
    allow_multiedges = data_model.get_graph_prior().with_parallel_edges()

    counter = 0
    for g in enumerate_all_graphs(size, edge_count, allow_self_loops, allow_multiedges):
        counter += 1
        if data_model.get_graph_prior().is_compatible(g):
            data_model.set_graph(g)
            likelihood = data_model.get_log_likelihood()
            prior = get_graph_log_evidence(data_model.get_graph_prior(), config)
            logevidence.append(prior + likelihood)

    data_model.set_graph(original_graph)
    return log_sum_exp(logevidence)


def get_log_evidence_meanfield(data_model: DynamicsWrapper, config: Config):
    log_joint = data_model.get_log_joint()
    log_posterior = get_log_posterior_meanfield(data_model, config)
    return log_joint - log_posterior


def get_log_evidence_exact_meanfield(data_model: DynamicsWrapper, config: Config):
    log_joint = data_model.get_log_joint()
    log_posterior = get_log_posterior_exact_meanfield(data_model, config)
    return log_joint - log_posterior


def get_log_evidence(data_model: DynamicsWrapper, config: Config = None, **kwargs):
    config = Config(**kwargs) if config is None else config
    method = config.get_value("method", "meanfield")
    functions = {
        "exact": get_log_evidence_exact,
        "exact_meanfield": get_log_evidence_exact_meanfield,
        "arithmetic": get_log_evidence_arithmetic,
        "harmonic": get_log_evidence_harmonic,
        "meanfield": get_log_evidence_meanfield,
        "full-meanfield": get_log_evidence_meanfield,
        "annealed": get_log_evidence_annealed,
    }
    if method in functions:
        return functions[method](data_model, config)
    else:
        message = (
            f"Invalid method {method}, valid methods" + f"are {list(functions.keys())}."
        )
        raise ValueError(message)


def get_log_posterior_arithmetic(data_model: DynamicsWrapper, config: Config):
    log_joint = data_model.get_log_joint()
    log_evidence = get_log_evidence_arithmetic(data_model, config)
    return log_joint - log_evidence


def get_log_posterior_harmonic(
    data_model: DynamicsWrapper, config: Config, verbose: int = 0
):
    log_joint = data_model.get_log_joint()
    log_evidence = get_log_evidence_harmonic(data_model, config, verbose=verbose)
    return log_joint - log_evidence


def get_log_posterior_annealed(data_model: DynamicsWrapper, config: Config):
    log_joint = data_model.get_log_joint()
    log_evidence = get_log_evidence_annealed(data_model, config)
    return log_joint - log_evidence


def get_log_posterior_meanfield(
    data_model: DynamicsWrapper, config: Config, verbose: int = 0
):
    graph_model = data_model.graph_prior
    mcmc = GraphReconstructionMCMC(data_model)
    original_graph = mcmc.get_graph()

    callback = CollectEdgesOnSweep(
        labeled=data_model.graph_prior.labeled, nested=data_model.graph_prior.nested
    )
    mcmc.insert_callback("collector", callback)
    callback.collect()
    if not config.get_value("start_from_original", False):
        mcmc.sample_prior()
    burn = config.burn_per_vertex * data_model.get_size()
    s, f = mcmc.do_MH_sweep(burn=config.initial_burn)

    # x = np.array(mcmc.get_data_model().get_past_states())

    # import matplotlib.pyplot as plt
    #
    # plt.plot(x.sum(0))
    # plt.show()

    for i in range(config.num_sweeps):
        t0 = time.time()
        _s, _f = mcmc.do_MH_sweep(burn=burn)
        t1 = time.time()
        if verbose:
            print(
                f"Sweep {i}:",
                f"time={t1 - t0}",
                f"successes={_s}",
                f"failures={burn - _s}",
                f"likelihood={data_model.get_log_likelihood()}",
                f"prior={mcmc.get_log_prior()}",
            )

    mcmc.set_graph(original_graph)
    logp = callback.get_log_posterior_estimate(original_graph)

    mcmc.remove_callback("collector")
    return logp


def get_log_posterior_annealed(
    data_model: DynamicsWrapper, config: Config, verbose: int = 0
):
    log_joint = data_model.get_log_joint()
    log_evidence = get_log_evidence_annealed(data_model, config, verbose)
    return log_joint - log_evidence


def get_log_posterior_exact(data_model: DynamicsWrapper, config: Config):
    log_joint = data_model.get_log_joint()
    log_evidence = get_log_evidence_exact(data_model, config)
    return log_joint - log_evidence


def get_log_posterior_exact_meanfield(data_model: DynamicsWrapper, config: Config):
    original_graph = data_model.get_graph()
    size = data_model.get_size()
    graph_prior = data_model.graph_prior
    edge_count = graph_prior.get_edge_count()
    allow_self_loops = data_model.get_graph_prior().with_self_loops()
    allow_multiedges = data_model.get_graph_prior().with_parallel_edges()

    i = 0
    edge_weights = defaultdict(lambda: defaultdict(list))
    edge_total = defaultdict(list)
    evidence = get_log_evidence_exact(data_model, config)
    for g in enumerate_all_graphs(size, edge_count, allow_self_loops, allow_multiedges):
        if graph_prior.is_compatible(g):
            i += 1
            data_model.set_graph(g)
            weight = data_model.get_log_joint() - evidence
            for e, w in get_weighted_edge_list(g).items():
                edge_weights[e][w].append(weight)
                edge_total[e].append(weight)

    data_model.set_graph(original_graph)
    log_posterior = 0
    for e, weights in edge_weights.items():
        probs = np.zeros(len(weights) + 1)
        probs[0] = 1 - np.exp(log_sum_exp(edge_total[e]))
        for w, ww in weights.items():
            probs[w] = np.exp(log_sum_exp(ww))
        probs = np.array(probs)
        w = original_graph.get_edge_multiplicity_idx(*e)
        log_posterior += np.log(probs[w])

    return log_posterior


def get_log_posterior(data_model: DynamicsWrapper, config: Config = None, **kwargs):
    config = Config(**kwargs) if config is None else config
    method = config.get_value("method", "meanfield")
    functions = {
        "exact": get_log_posterior_exact,
        "exact_meanfield": get_log_posterior_exact_meanfield,
        "arithmetic": get_log_posterior_arithmetic,
        "harmonic": get_log_posterior_harmonic,
        "meanfield": get_log_posterior_meanfield,
        "full-meanfield": get_log_posterior_meanfield,
        "annealed": get_log_posterior_annealed,
    }
    if method in functions:
        return functions[method](data_model, config)
    else:
        message = (
            f"Invalid method {method}, valid methods" + f"are {list(functions.keys())}."
        )
        raise ValueError(message)


def get_log_prior_meanfield(
    data_model: DynamicsWrapper, config: Config, verbose: int = 0
) -> float:
    mcmc = GraphReconstructionMCMC(data_model, verbose=verbose)
    callback = CollectEdgesOnSweep(
        labeled=data_model.graph_prior.labeled, nested=data_model.graph_prior.nested
    )
    mcmc.insert_callback("edge", callback)

    randomGraph = mcmc.get_graph_prior()
    original_graph = mcmc.get_graph()
    callback.collect()
    for i in range(config.num_sweeps):
        randomGraph.sample()
        callback.collect()
    hg = callback.get_log_posterior_estimate(original_graph)
    mcmc.set_graph(original_graph)
    mcmc.remove_callback("edge")
    return hg


def get_graph_log_evidence_meanfield(graph_model: RandomGraphWrapper, config: Config):
    if importlib.util.find_spec("graph_tool"):
        from graph_tool.inference import ModeClusterState, mcmc_equilibrate
    else:
        import warnings

        warnings.warn("`graph_tool` has not been found, proceeding anyway.")
        return 0.0
    og_p = graph_model.get_labels()
    og_g = graph_model.get_state()
    burn = config.get_value("burn_per_vertex", 10) * graph_model.get_size()

    mcmc = PartitionReconstructionMCMC(graph_model)
    if not config.get_value("start_from_original", True):
        graph_model.sample()
        graph_model.set_state(og_g)

    _, _ = mcmc.do_MH_sweep(burn=config.get_value("initial_burn", burn))

    callback = CollectPartitionOnSweep(nested=graph_model.nested)
    mcmc.insert_callback("partitions", callback)

    for i in range(config.get_value("num_sweeps", 100)):
        _s, _f = mcmc.do_MH_sweep(burn=burn)

    partitions = callback.get_data()
    pmodes = ModeClusterState(partitions, nested=graph_model.nested)  # from graph-tool
    if config.get_value("equilibrate_mode_cluster", False):
        mcmc_equilibrate(pmodes, force_niter=10, verbose=True)
        # for i in range(config.get_value("num_sweeps", 100)):
        #     print(i, pmodes.entropy(), pmodes.mcmc_sweep())
    samples = []
    for p in partitions:
        graph_model.set_labels(p)
        samples.append(graph_model.get_log_joint() + loggamma(1 + len(np.unique(p))))

    log_evidence = np.mean(samples) + pmodes.posterior_entropy()
    graph_model.set_labels(og_p)
    return log_evidence


def get_graph_log_evidence_exact(
    graph_model: RandomGraphWrapper, config: Config
) -> float:
    if graph_model.nested:
        raise TypeError("`graph_model` must not be nested.")

    logp = []
    og_p = graph_model.get_labels()
    for p in enumerate_all_partitions(
        graph_model.get_size(), graph_model.get_size() - 1
    ):
        graph_model.set_labels(p, False)
        logp.append(graph_model.get_log_joint())

    log_evidence = log_sum_exp(logp)
    graph_model.set_labels(og_p)
    return log_evidence


def get_graph_log_evidence_annealed(
    graph_model: RandomGraphWrapper, config: Config, verbose=False
) -> float:

    mcmc = PartitionReconstructionMCMC(graph_model)
    callback = CollectLikelihoodOnSweep()
    mcmc.insert_callback("likelihoods", callback)
    og_p = mcmc.get_labels()

    burn = config.get_value("burn_per_vertex", 10) * graph_model.get_size()
    logp = []

    beta_k = np.linspace(0, 1, config.get_value("num_betas", 100) + 1) ** (
        1 / config.get_value("exp_betas", 0.5)
    )
    for lb, ub in zip(beta_k[:-1], beta_k[1:]):
        if verbose:
            print(f"beta: {lb}")
        mcmc.set_beta_likelihood(lb)
        if config.get_value("start_from_original", True):
            mcmc.set_labels(og_p)
        else:
            mcmc.sample_prior()
        s, f = mcmc.do_MH_sweep(burn=config.get_value("initial_burn", burn))

        for i in range(config.get_value("num_sweeps", 1000)):
            mcmc.do_MH_sweep(burn=burn)
        logp_k = (ub - lb) * np.array(callback.get_data())
        logp.append(log_mean_exp(logp_k))
        callback.clear()

    log_evidence = sum(logp)
    graph_model.set_labels(og_p)
    return log_evidence


def get_graph_log_evidence(
    graph_model: RandomGraphWrapper, config: Config = None, **kwargs
) -> float:
    config = Config(**kwargs) if config is None else config

    method = config.get_value("method", "meanfield")

    if not graph_model.labeled:
        return graph_model.get_log_joint()
    functions = {
        "exact": get_graph_log_evidence_exact,
        "meanfield": get_graph_log_evidence_meanfield,
        "annealed": get_graph_log_evidence_annealed,
    }
    if method in functions:
        return functions[method](graph_model, config)
    else:
        message = (
            f"Invalid method {method}, valid methods are {list(functions.keys())}."
        )
        raise ValueError(message)
