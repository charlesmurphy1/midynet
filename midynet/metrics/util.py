import numpy as np
from collections import defaultdict
from _midynet.mcmc import DynamicsMCMC
from _midynet.mcmc.callbacks import (
    CollectEdgeMultiplicityOnSweep,
    CollectLikelihoodOnSweep,
)
from _midynet.utility import get_weighted_edge_list
from midynet.config import Config, MCMCVerboseFactory
from midynet.util import enumerate_all_graphs, log_mean_exp, log_sum_exp

__all__ = ("get_log_evidence", "get_log_posterior")


def get_log_evidence_arithmetic(dynamicsMCMC: DynamicsMCMC, config: Config, **kwargs):
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


def get_log_evidence_harmonic(
    dynamicsMCMC: DynamicsMCMC, config: Config, verbose: int = 0, **kwargs
):
    callback = CollectLikelihoodOnSweep()
    dynamicsMCMC.insert_callback("collector", callback)

    verboseCallback = MCMCVerboseFactory.build_console()
    if verbose:
        dynamicsMCMC.insert_callback("verbose", verboseCallback.get_wrap())

    g = dynamicsMCMC.get_graph()
    dynamicsMCMC.set_up()
    burn = config.burn_per_vertex * dynamicsMCMC.get_dynamics().get_size()
    s, f = dynamicsMCMC.do_MH_sweep(burn=config.initial_burn)
    for i in range(config.num_sweeps):
        s, f = dynamicsMCMC.do_MH_sweep(burn=burn)

    logp = -np.array(callback.get_log_likelihoods())

    dynamicsMCMC.tear_down()
    dynamicsMCMC.remove_callback("collector")
    if verbose:
        dynamicsMCMC.remove_callback("verbose")
    dynamicsMCMC.set_graph(g)
    return log_mean_exp(logp)


def get_log_evidence_meanfield(
    dynamicsMCMC: DynamicsMCMC, config: Config, verbose: int = 0, **kwargs
):

    return dynamicsMCMC.get_log_joint() - get_log_posterior_meanfield(
        dynamicsMCMC, config, verbose, **kwargs
    )


def get_log_evidence_annealed(
    dynamicsMCMC: DynamicsMCMC, config: Config, verbose: int = 0, **kwargs
):
    callback = CollectLikelihoodOnSweep()
    dynamicsMCMC.insert_callback("collector", callback)
    verboseCallback = MCMCVerboseFactory.build_console()
    if verbose:
        dynamicsMCMC.insert_callback("verbose", verboseCallback.get_wrap())

    original_graph = dynamicsMCMC.get_graph()
    dynamicsMCMC.set_up()
    burn = config.burn_per_vertex * dynamicsMCMC.get_dynamics().get_size()
    logp = []

    beta_k = np.linspace(0, 1, config.num_betas + 1) ** (1 / config.exp_betas)
    for lb, ub in zip(beta_k[:-1], beta_k[1:]):
        if verbose:
            print(f"beta: {lb}")
        dynamicsMCMC.set_beta_likelihood(lb)
        if config.start_from_original:
            dynamicsMCMC.set_graph(original_graph)
        else:
            dynamicsMCMC.get_dynamics().sample_graph()
        dynamicsMCMC.set_up()

        # if config.start_from_original:
        #     dynamicsMCMC.set_graph(original_graph)
        # else:
        #     dynamicsMCMC.get_dynamics().sample_graph()
        #     dynamicsMCMC.set_graph(dynamicsMCMC.get_graph())
        s, f = dynamicsMCMC.do_MH_sweep(burn=config.initial_burn)
        for i in range(config.num_sweeps):
            dynamicsMCMC.do_MH_sweep(burn=burn)
            dynamicsMCMC.check_consistency()
        logp_k = (ub - lb) * np.array(callback.get_log_likelihoods())
        logp.append(log_mean_exp(logp_k))
        callback.clear()
    dynamicsMCMC.tear_down()
    dynamicsMCMC.remove_callback("collector")
    if verbose:
        dynamicsMCMC.remove_callback("verbose")
    dynamicsMCMC.set_graph(original_graph)

    return sum(logp)


def get_log_evidence_exact(dynamicsMCMC: DynamicsMCMC, config: Config, **kwargs):
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
    for g in enumerate_all_graphs(size, edge_count, allow_self_loops, allow_multiedges):
        counter += 1
        if graph.is_compatible(g):
            dynamicsMCMC.set_graph(g)
            logevidence.append(dynamicsMCMC.get_log_joint())
    dynamicsMCMC.tear_down()

    dynamicsMCMC.set_graph(original_graph)
    return log_sum_exp(logevidence)


def get_log_evidence_exact_meanfield(
    dynamicsMCMC: DynamicsMCMC, config: Config, **kwargs
):
    return dynamicsMCMC.get_log_joint() - get_log_posterior_exact_meanfield(
        dynamicsMCMC, config
    )


def get_log_evidence(dynamicsMCMC: DynamicsMCMC, config: Config, **kwargs):
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
        return functions[method](dynamicsMCMC, config, **kwargs)
    else:
        message = (
            f"Invalid method {method}, valid methods" + f"are {list(functions.keys())}."
        )
        raise ValueError(message)


def get_log_posterior_arithmetic(dynamicsMCMC: DynamicsMCMC, config: Config, **kwargs):
    return dynamicsMCMC.get_log_joint() - get_log_evidence_arithmetic(
        dynamicsMCMC, config, **kwargs
    )


def get_log_posterior_harmonic(dynamicsMCMC: DynamicsMCMC, config: Config, **kwargs):
    return dynamicsMCMC.get_log_joint() - get_log_evidence_harmonic(
        dynamicsMCMC, config, **kwargs
    )


def get_log_posterior_meanfield(
    dynamicsMCMC: DynamicsMCMC, config: Config, verbose: int = 0, **kwargs
):
    graph_callback = CollectEdgeMultiplicityOnSweep()
    dynamicsMCMC.insert_callback("collector", graph_callback)
    if verbose:
        verboseCallback = MCMCVerboseFactory.build_console()
        dynamicsMCMC.insert_callback("verbose", verboseCallback.get_wrap())
    original_graph = dynamicsMCMC.get_graph()
    graph_callback.collect()
    # if not config.start_from_original:
    #     dynamicsMCMC.get_dynamics().sample_graph()
    #     dynamicsMCMC.set_graph(dynamicsMCMC.get_graph())
    # if not config.start_from_original:
    #     dynamicsMCMC.get_dynamics().sample_graph()
    #     dynamicsMCMC.set_up()
    burn = config.burn_per_vertex * dynamicsMCMC.get_dynamics().get_size()
    s, f = dynamicsMCMC.do_MH_sweep(burn=config.initial_burn)

    for i in range(config.num_sweeps):
        _s, _f = dynamicsMCMC.do_MH_sweep(burn=burn)
        s += _s
        f += _f

    dynamicsMCMC.set_graph(original_graph)
    logp = graph_callback.get_log_posterior_estimate(original_graph)

    dynamicsMCMC.tear_down()
    dynamicsMCMC.remove_callback("collector")
    if verbose:
        dynamicsMCMC.remove_callback("verbose")
    return logp


def get_log_posterior_annealed(dynamicsMCMC: DynamicsMCMC, config: Config, **kwargs):
    return dynamicsMCMC.get_log_joint() - get_log_evidence_annealed(
        dynamicsMCMC, config, **kwargs
    )


def get_log_posterior_exact(dynamicsMCMC: DynamicsMCMC, config: Config, **kwargs):
    return dynamicsMCMC.get_log_joint() - get_log_evidence_exact(
        dynamicsMCMC, config, **kwargs
    )


def get_log_posterior_exact_meanfield(
    dynamicsMCMC: DynamicsMCMC, config: Config, **kwargs
):
    original_graph = dynamicsMCMC.get_graph()
    size = dynamicsMCMC.get_dynamics().get_size()
    edge_proposer = dynamicsMCMC.get_random_graph_mcmc().get_edge_proposer()
    graph = dynamicsMCMC.get_random_graph_mcmc().get_random_graph()
    edge_count = graph.get_edge_count()
    allow_self_loops = edge_proposer.allow_self_loops()
    allow_multiedges = edge_proposer.allow_multiedges()

    dynamicsMCMC.set_up()
    i = 0
    edge_weights = defaultdict(lambda: defaultdict(list))
    edge_total = defaultdict(list)
    evidence = get_log_evidence_exact(dynamicsMCMC, config)
    for g in enumerate_all_graphs(size, edge_count, allow_self_loops, allow_multiedges):
        if graph.is_compatible(g):
            i += 1
            dynamicsMCMC.get_dynamics().set_graph(g)
            weight = dynamicsMCMC.get_log_joint() - evidence
            for e, w in get_weighted_edge_list(g).items():
                edge_weights[e][w].append(weight)
                edge_total[e].append(weight)
    dynamicsMCMC.tear_down()
    dynamicsMCMC.set_graph(original_graph)
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


def get_log_posterior(dynamicsMCMC: DynamicsMCMC, config: Config, **kwargs):
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
        return functions[method](dynamicsMCMC, config, **kwargs)
    else:
        message = (
            f"Invalid method {method}, valid methods" + f"are {list(functions.keys())}."
        )
        raise ValueError(message)


def get_log_prior_meanfield(
    dynamicsMCMC: DynamicsMCMC, config: Config, **kwargs
) -> float:
    callback = CollectEdgeMultiplicityOnSweep()
    dynamicsMCMC.insert_callback("collector", callback)
    randomGraph = dynamicsMCMC.get_random_graph_mcmc().get_random_graph()
    original_graph = dynamicsMCMC.get_graph()
    callback.collect()
    for i in range(config.num_sweeps):
        randomGraph.sample()
        callback.collect()
    hg = callback.get_log_posterior_estimate(original_graph)
    dynamicsMCMC.set_graph(original_graph)
    dynamicsMCMC.remove_callback("collector")
    return hg
