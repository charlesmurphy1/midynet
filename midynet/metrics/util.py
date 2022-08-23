from __future__ import annotations
import numpy as np
from collections import defaultdict
from graph_tool.inference import ModeClusterState, mcmc_equilibrate

from midynet.random_graph import (
    BlockLabeledRandomGraph,
    NestedBlockLabeledRandomGraph,
)
from midynet.mcmc import GraphReconstructionMCMC, PartitionReconstructionMCMC
from midynet.mcmc.callbacks import (
    CollectEdgesOnSweep,
    CollectPartitionOnSweep,
    CollectLikelihoodOnSweep,
)
from midynet.config import Config
from midynet.utility import (
    get_weighted_edge_list,
    enumerate_all_graphs,
    enumerate_all_partitions,
    log_mean_exp,
    log_sum_exp,
)

__all__ = (
    "get_log_evidence",
    "get_log_posterior",
    "get_posterior_entropy_partition",
)


def get_log_evidence_arithmetic(data_model: DynamicsWrapper, config: Config, **kwargs):
    mcmc = GraphReconstructionMCMC(data_model, verbose=kwargs.get("verbose", 0))
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
    data_model: DynamicsWrapper, config: Config, verbose: int = 0, **kwargs
):
    mcmc = GraphReconstructionMCMC(data_model, verbose=kwargs.get("verbose", 0))
    callback = CollectLikelihoodOnSweep()
    mcmc.insert_callback("collector", callback)

    g = mcmc.get_graph()
    burn = config.burn_per_vertex * mcmc.get_data_model().get_size()
    s, f = mcmc.do_MH_sweep(burn=config.initial_burn)
    for i in range(config.num_sweeps):
        s, f = mcmc.do_MH_sweep(burn=burn)

    logp = -np.array(callback.get_data())

    mcmc.remove_callback("collector")
    if verbose:
        mcmc.remove_callback("verbose")
    mcmc.set_graph(g)
    return log_mean_exp(logp)


def get_log_evidence_meanfield(
    data_model: DynamicsWrapper, config: Config, verbose: int = 0, **kwargs
):

    logJoint = data_model.get_log_joint()
    logPosterior = get_log_posterior_meanfield(data_model, config, verbose, **kwargs)
    S = logJoint - logPosterior
    print(S, logJoint, "-", logPosterior)
    if data_model.graph_prior.labeled:
        S -= get_posterior_entropy_partition_meanfield(
            data_model.get_graph_prior(), config
        )
    return S


def get_log_evidence_annealed(
    data_model: DynamicsWrapper, config: Config, verbose: int = 0, **kwargs
):
    mcmc = GraphReconstructionMCMC(data_model, verbose=kwargs.get("verbose", 0))
    callback = CollectLikelihoodOnSweep()
    mcmc.insert_callback("collector", callback)
    original_graph = mcmc.get_graph()

    burn = config.burn_per_vertex * mcmc.get_data_model().get_size()
    logp = []

    beta_k = np.linspace(0, 1, config.num_betas + 1) ** (1 / config.exp_betas)
    for lb, ub in zip(beta_k[:-1], beta_k[1:]):
        if verbose:
            print(f"beta: {lb}")
        mcmc.set_beta_likelihood(lb)
        if config.start_from_original:
            mcmc.set_graph(original_graph)
        else:
            mcmc.sample_prior()
        s, f = mcmc.do_MH_sweep(burn=config.initial_burn)
        for i in range(config.num_sweeps):
            mcmc.do_MH_sweep(burn=burn)
            mcmc.check_consistency()
        logp_k = (ub - lb) * np.array(callback.get_data())
        logp.append(log_mean_exp(logp_k))
        callback.clear()

    mcmc.remove_callback("collector")
    if verbose:
        mcmc.remove_callback("verbose")
    mcmc.set_graph(original_graph)

    return sum(logp)


def get_log_evidence_exact(data_model: DynamicsWrapper, config: Config, **kwargs):
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
            logevidence.append(data_model.get_log_joint())

    data_model.set_graph(original_graph)
    return log_sum_exp(logevidence)


def get_log_evidence_exact_meanfield(
    data_model: DynamicsWrapper, config: Config, **kwargs
):
    S = data_model.get_log_joint() - get_log_posterior_exact_meanfield(
        data_model, config
    )
    if data_model.graph_prior.labeled:
        S -= get_log_posterior_meanfield_partition(data_model.graph_prior, config)
    return S


def get_log_evidence(data_model: DynamicsWrapper, config: Config, **kwargs):
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
        return functions[method](data_model, config, **kwargs)
    else:
        message = (
            f"Invalid method {method}, valid methods" + f"are {list(functions.keys())}."
        )
        raise ValueError(message)


def get_log_posterior_arithmetic(data_model: DynamicsWrapper, config: Config, **kwargs):
    S = data_model.get_log_joint() - get_log_evidence_arithmetic(
        data_model, config, **kwargs
    )
    if data_model.graph_prior.labeled:
        S -= get_log_posterior_meanfield_partition(data_model.get_graph_prior(), config)
    return S


def get_log_posterior_harmonic(data_model: DynamicsWrapper, config: Config, **kwargs):
    S = data_model.get_log_joint() - get_log_evidence_harmonic(
        data_model, config, **kwargs
    )
    if data_model.graph_prior.labeled:
        S -= get_log_posterior_meanfield_partition(data_model.graph_prior, config)
    return S


def get_log_posterior_meanfield(
    data_model: DynamicsWrapper, config: Config, verbose: int = 0, **kwargs
):
    mcmc = GraphReconstructionMCMC(data_model, verbose=kwargs.get("verbose", 0))
    original_graph = mcmc.get_graph()

    callback = CollectEdgesOnSweep(
        labeled=data_model.graph_prior.labeled, nested=data_model.graph_prior.nested
    )
    mcmc.insert_callback("collector", callback)
    callback.collect()
    callback.collect()
    if not config.start_from_original:
        mcmc.sample_prior()
    burn = config.burn_per_vertex * mcmc.get_data_model().get_size()
    s, f = mcmc.do_MH_sweep(burn=config.initial_burn)

    for i in range(config.num_sweeps):
        _s, _f = mcmc.do_MH_sweep(burn=burn)
        print(i, _s, _f)
        s += _s
        f += _f

    mcmc.set_graph(original_graph)
    logp = callback.get_log_posterior_estimate(original_graph)

    mcmc.remove_callback("collector")
    return logp


def get_log_posterior_annealed(data_model: DynamicsWrapper, config: Config, **kwargs):
    S = data_model.get_log_joint() - get_log_evidence_annealed(
        data_model, config, **kwargs
    )
    if issubclass(data_model.get_graph_prior().__class__, BlockLabeledRandomGraph):
        S -= get_log_posterior_meanfield_partition(data_model.get_graph_prior(), config)
    return S


def get_log_posterior_exact(data_model: DynamicsWrapper, config: Config, **kwargs):
    S = data_model.get_log_joint() - get_log_evidence_exact(
        data_model, config, **kwargs
    )
    if issubclass(data_model.get_graph_prior().__class__, BlockLabeledRandomGraph):
        S -= get_log_posterior_exact_partition(data_model.get_graph_prior(), config)
    return S


def get_log_posterior_exact_meanfield(
    data_model: DynamicsWrapper, config: Config, **kwargs
):
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


def get_log_posterior(data_model: DynamicsWrapper, config: Config, **kwargs):
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
        return functions[method](data_model, config, **kwargs)
    else:
        message = (
            f"Invalid method {method}, valid methods" + f"are {list(functions.keys())}."
        )
        raise ValueError(message)


def get_log_prior_meanfield(
    data_model: DynamicsWrapper, config: Config, **kwargs
) -> float:
    mcmc = GraphReconstructionMCMC(data_model, verbose=kwargs.get("verbose", 0))
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


def get_posterior_entropy_partition_meanfield(
    graph_model: BlockLabeledRandomGraph, config: Config = None, **kwargs
) -> float:
    config = Config(**kwargs) if config is None else config
    print(config)
    mcmc = PartitionReconstructionMCMC(graph_model)
    callback = CollectPartitionOnSweep()
    mcmc.insert_callback("partition", callback)
    original_partition = mcmc.get_labels()
    burn = config.burn_per_vertex * graph_model.get_size()

    for i in range(config.num_sweeps):
        _s, _f = mcmc.do_MH_sweep(burn=burn)
        print(graph_model.get_labels())
    partitions = callback.get_data()
    print(partitions)
    pmodes = ModeClusterState(partitions)  # from graph-tool
    if config.get_value("equilibrate_mode_cluster", False):
        mcmc_equilibrate(pmodes, force_niter=10, verbose=True)  # from graph-tool
    S = pmodes.posterior_entropy(True)
    graph_model.set_labels(original_partition)
    return S


def get_posterior_entropy_partition_exact(
    graph_model: RandomGraph, config: Config = None, **kwargs
) -> float:

    config = Config(**kwargs) if config is None else config
    logp = []
    og_p = graph_model.get_labels()
    for p in enumerate_all_partitions(graph_model.get_size(), graph_model.get_size()):
        graph_model.set_labels(p)
        logp.append(graph_model.get_log_joint())

    log_evidence = log_sum_exp(logp)

    entropy = 0
    z = 0
    for p in enumerate_all_partitions(graph_model.get_size(), graph_model.get_size()):
        graph_model.set_labels(p)
        log_posterior = graph_model.get_log_joint() - log_evidence
        z += np.exp(log_posterior)
        print(p, graph_model.get_labels(), log_posterior, graph_model.get_label_count())
        if not np.isinf(log_posterior):
            entropy -= np.exp(log_posterior) * log_posterior
    print(f"{entropy=}, {z=}")
    graph_model.set_labels(og_p)
    return entropy


def get_posterior_entropy_partition(
    graph_model: RandomGraph, config: Config, **kwargs
) -> float:
    method = config.get_value("method", "meanfield")
    functions = {
        "exact": get_posterior_entropy_partition_exact,
        "meanfield": get_posterior_entropy_partition_meanfield,
        "annealed": get_posterior_entropy_partition_meanfield,
        "arithmetic": get_posterior_entropy_partition_meanfield,
        "harmonic": get_posterior_entropy_partition_meanfield,
    }
    if method in functions:
        return functions[method](graph_model, config, **kwargs)
    else:
        message = (
            f"Invalid method {method}, valid methods are {list(functions.keys())}."
        )
        raise ValueError(message)
