import numpy as np
from collections import defaultdict
from _midynet.mcmc import GraphReconstructionMCMC
from _midynet.mcmc.callbacks import (
    CollectEdgeMultiplicityOnSweep,
    CollectLikelihoodOnSweep,
)
from _midynet.utility import get_weighted_edge_list
from midynet.config import Config, MCMCVerboseFactory
from midynet.util import enumerate_all_graphs, log_mean_exp, log_sum_exp

__all__ = ("get_log_evidence", "get_log_posterior")


def get_log_evidence_arithmetic(
    mcmc: GraphReconstructionMCMC, config: Config, **kwargs
):
    logp = []
    g = mcmc.get_graph()
    for k in range(config.K):
        logp_k = []
        for m in range(config.num_sweeps):
            mcmc.get_dynamics().sample_graph()
            logp_k.append(mcmc.get_log_likelihood())
        logp.append(log_mean_exp(logp_k))
    mcmc.set_graph(g)

    return np.mean(logp)


def get_log_evidence_harmonic(
    mcmc: GraphReconstructionMCMC, config: Config, verbose: int = 0, **kwargs
):
    callback = CollectLikelihoodOnSweep()
    mcmc.insert_callback("collector", callback)

    verboseCallback = MCMCVerboseFactory.build_console()
    if verbose:
        mcmc.insert_callback("verbose", verboseCallback.get_wrap())

    g = mcmc.get_graph()
    mcmc.set_up()
    burn = config.burn_per_vertex * mcmc.get_dynamics().get_size()
    s, f = mcmc.do_MH_sweep(burn=config.initial_burn)
    for i in range(config.num_sweeps):
        s, f = mcmc.do_MH_sweep(burn=burn)

    logp = -np.array(callback.get_log_likelihoods())

    mcmc.tear_down()
    mcmc.remove_callback("collector")
    if verbose:
        mcmc.remove_callback("verbose")
    mcmc.set_graph(g)
    return log_mean_exp(logp)


def get_log_evidence_meanfield(
    mcmc: GraphReconstructionMCMC, config: Config, verbose: int = 0, **kwargs
):

    return mcmc.get_log_joint() - get_log_posterior_meanfield(
        mcmc, config, verbose, **kwargs
    )


def get_log_evidence_annealed(
    mcmc: GraphReconstructionMCMC, config: Config, verbose: int = 0, **kwargs
):
    callback = CollectLikelihoodOnSweep()
    mcmc.insert_callback("collector", callback)
    verboseCallback = MCMCVerboseFactory.build_console()
    if verbose:
        mcmc.insert_callback("verbose", verboseCallback.get_wrap())

    original_graph = mcmc.get_graph()
    mcmc.set_up()
    burn = config.burn_per_vertex * mcmc.get_dynamics().get_size()
    logp = []

    beta_k = np.linspace(0, 1, config.num_betas + 1) ** (1 / config.exp_betas)
    for lb, ub in zip(beta_k[:-1], beta_k[1:]):
        if verbose:
            print(f"beta: {lb}")
        mcmc.set_beta_likelihood(lb)
        if config.start_from_original:
            mcmc.set_graph(original_graph)
        else:
            mcmc.get_dynamics().sample_graph()
        mcmc.set_up()

        # if config.start_from_original:
        #     mcmc.set_graph(original_graph)
        # else:
        #     mcmc.get_dynamics().sample_graph()
        #     mcmc.set_graph(mcmc.get_graph())
        s, f = mcmc.do_MH_sweep(burn=config.initial_burn)
        for i in range(config.num_sweeps):
            mcmc.do_MH_sweep(burn=burn)
            mcmc.check_consistency()
        logp_k = (ub - lb) * np.array(callback.get_log_likelihoods())
        logp.append(log_mean_exp(logp_k))
        callback.clear()
    mcmc.tear_down()
    mcmc.remove_callback("collector")
    if verbose:
        mcmc.remove_callback("verbose")
    mcmc.set_graph(original_graph)

    return sum(logp)


def get_log_evidence_exact(mcmc: GraphReconstructionMCMC, config: Config, **kwargs):
    logevidence = []
    original_graph = mcmc.get_graph()
    size = mcmc.get_dynamics().get_size()
    edge_proposer = mcmc.get_edge_proposer()
    graph = mcmc.get_graph_prior()
    edge_count = graph.get_edge_count()
    allow_self_loops = edge_proposer.allow_self_loops()
    allow_multiedges = edge_proposer.allow_multiedges()

    mcmc.set_up()
    counter = 0
    for g in enumerate_all_graphs(size, edge_count, allow_self_loops, allow_multiedges):
        counter += 1
        if graph.is_compatible(g):
            mcmc.set_graph(g)
            logevidence.append(mcmc.get_log_joint())
    mcmc.tear_down()

    mcmc.set_graph(original_graph)
    return log_sum_exp(logevidence)


def get_log_evidence_exact_meanfield(
    mcmc: GraphReconstructionMCMC, config: Config, **kwargs
):
    return mcmc.get_log_joint() - get_log_posterior_exact_meanfield(mcmc, config)


def get_log_evidence(mcmc: GraphReconstructionMCMC, config: Config, **kwargs):
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
        return functions[method](mcmc, config, **kwargs)
    else:
        message = (
            f"Invalid method {method}, valid methods" + f"are {list(functions.keys())}."
        )
        raise ValueError(message)


def get_log_posterior_arithmetic(
    mcmc: GraphReconstructionMCMC, config: Config, **kwargs
):
    return mcmc.get_log_joint() - get_log_evidence_arithmetic(mcmc, config, **kwargs)


def get_log_posterior_harmonic(mcmc: GraphReconstructionMCMC, config: Config, **kwargs):
    return mcmc.get_log_joint() - get_log_evidence_harmonic(mcmc, config, **kwargs)


def get_log_posterior_meanfield(
    mcmc: GraphReconstructionMCMC, config: Config, verbose: int = 0, **kwargs
):
    graph_callback = CollectEdgeMultiplicityOnSweep()
    mcmc.insert_callback("collector", graph_callback)
    if verbose:
        verboseCallback = MCMCVerboseFactory.build_console()
        mcmc.insert_callback("verbose", verboseCallback.get_wrap())
    mcmc.set_up()
    original_graph = mcmc.get_graph()
    graph_callback.collect()
    # if not config.start_from_original:
    #     mcmc.get_dynamics().sample_graph()
    #     mcmc.set_graph(mcmc.get_graph())
    # if not config.start_from_original:
    #     mcmc.get_dynamics().sample_graph()
    #     mcmc.set_up()
    burn = config.burn_per_vertex * mcmc.get_dynamics().get_size()
    s, f = mcmc.do_MH_sweep(burn=config.initial_burn)

    for i in range(config.num_sweeps):
        _s, _f = mcmc.do_MH_sweep(burn=burn)
        s += _s
        f += _f

    mcmc.set_graph(original_graph)
    logp = graph_callback.get_log_posterior_estimate(original_graph)

    mcmc.tear_down()
    mcmc.remove_callback("collector")
    if verbose:
        mcmc.remove_callback("verbose")
    return logp


def get_log_posterior_annealed(mcmc: GraphReconstructionMCMC, config: Config, **kwargs):
    return mcmc.get_log_joint() - get_log_evidence_annealed(mcmc, config, **kwargs)


def get_log_posterior_exact(mcmc: GraphReconstructionMCMC, config: Config, **kwargs):
    return mcmc.get_log_joint() - get_log_evidence_exact(mcmc, config, **kwargs)


def get_log_posterior_exact_meanfield(
    mcmc: GraphReconstructionMCMC, config: Config, **kwargs
):
    original_graph = mcmc.get_graph()
    size = mcmc.get_dynamics().get_size()
    edge_proposer = mcmc.get_edge_proposer()
    graph = mcmc.get_graph_prior()
    edge_count = graph.get_edge_count()
    allow_self_loops = edge_proposer.allow_self_loops()
    allow_multiedges = edge_proposer.allow_multiedges()

    mcmc.set_up()
    i = 0
    edge_weights = defaultdict(lambda: defaultdict(list))
    edge_total = defaultdict(list)
    evidence = get_log_evidence_exact(mcmc, config)
    for g in enumerate_all_graphs(size, edge_count, allow_self_loops, allow_multiedges):
        if graph.is_compatible(g):
            i += 1
            mcmc.get_dynamics().set_graph(g)
            weight = mcmc.get_log_joint() - evidence
            for e, w in get_weighted_edge_list(g).items():
                edge_weights[e][w].append(weight)
                edge_total[e].append(weight)
    mcmc.tear_down()
    mcmc.set_graph(original_graph)
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


def get_log_posterior(mcmc: GraphReconstructionMCMC, config: Config, **kwargs):
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
        return functions[method](mcmc, config, **kwargs)
    else:
        message = (
            f"Invalid method {method}, valid methods" + f"are {list(functions.keys())}."
        )
        raise ValueError(message)


def get_log_prior_meanfield(
    mcmc: GraphReconstructionMCMC, config: Config, **kwargs
) -> float:
    callback = CollectEdgeMultiplicityOnSweep()
    mcmc.insert_callback("edge", callback)
    mcmc.set_up()
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
