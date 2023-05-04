from __future__ import annotations
from collections import defaultdict
from scipy.special import loggamma
import numpy as np
import importlib
import time

from basegraph import core
from graphinf.data import DataModelWrapper
from graphinf.graph import RandomGraphWrapper
import graphinf

from pyhectiqlab import Config
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


def data_model_mcmc_sweep(
    model: DataModelWrapper,
    num_steps: int,
    sweep_type: str = "metropolis",
    graph_rate: float = 1,
    prior_rate: float = 0,
    param_rate: float = 0,
    beta_prior: float = 1,
    beta_likelihood: float = 1,
    **kwargs,
) -> int:
    if sweep_type == "metropolis":
        z = graph_rate + prior_rate + param_rate
        graph_rate /= z
        prior_rate /= z
        param_rate /= z
        return model.metrololis_sweep(
            num_steps,
            graph_rate=graph_rate,
            prior_rate=prior_rate,
            param_rate=param_rate,
            beta_prior=beta_prior,
            beta_likelihood=beta_likelihood,
        )


def get_log_evidence_arithmetic(data_model: DataModelWrapper, config: Config):
    logp = []
    g = data_model.get_graph()
    for _ in range(config.K):
        samples = []
        for _ in range(config.num_sweeps):
            data_model.sample_prior()
            samples.append(data_model.get_log_likelihood())
        logp.append(log_mean_exp(samples))
    data_model.set_graph(g)

    return np.mean(logp)


def get_log_evidence_harmonic(
    data_model: DataModelWrapper, config: Config, verbose: int = 0
):
    burn = config.burn_per_vertex * data_model.get_size()
    g = data_model.get_graph()

    s = data_model_mcmc_sweep(num_steps=config.initial_burn, **config)
    likelihoods = []
    for i in range(config.num_sweeps):
        s = data_model_mcmc_sweep(data_model, num_steps=burn, **config)
        likelihoods.append(-data_model.get_log_likelihood())

    if config.get("reset_graph", True):
        data_model.set_graph(g)
    return log_mean_exp(likelihoods)


def get_log_evidence_annealed(
    data_model: DataModelWrapper, config: Config, verbose: int = 0
):
    original_graph = data_model.get_graph()
    burn = config.burn_per_vertex * data_model.get_size()
    logp = []

    beta_k = np.linspace(0, 1, config.num_betas + 1) ** (1 / config.exp_betas)
    for lb, ub in zip(beta_k[:-1], beta_k[1:]):
        if verbose:
            print(f"beta: {lb}")
        if config.get("start_from_original", False):
            data_model.set_graph(original_graph)
        else:
            data_model.sample_prior()
        likelihoods = []
        data_model_mcmc_sweep(
            data_model, num_steps=config.initial_burn, beta_likelihood=lb, **config
        )
        for i in range(config.num_sweeps):
            data_model_mcmc_sweep(
                data_model, num_steps=burn, beta_likelihood=lb, **config
            )
            likelihoods.append(data_model.get_log_likelihood())
        logp_k = (ub - lb) * np.array(likelihoods)
        logp.append(log_mean_exp(logp_k))

    if config.get("reset_graph", True):
        data_model.set_graph(original_graph)

    return sum(logp)


def get_log_evidence_exact(data_model: DataModelWrapper, config: Config):
    logevidence = []
    original_graph = data_model.get_graph()
    size = data_model.get_size()
    edge_count = data_model.get_graph().get_total_edge_number()
    allow_self_loops = data_model.graph_prior.with_self_loops()
    allow_multiedges = data_model.graph_prior.with_parallel_edges()

    counter = 0
    for g in enumerate_all_graphs(size, edge_count, allow_self_loops, allow_multiedges):
        counter += 1
        if data_model.graph_prior.is_compatible(g):
            data_model.set_graph(g)
            likelihood = data_model.get_log_likelihood()
            prior = get_graph_log_evidence(data_model.get_graph_prior(), config)
            logevidence.append(prior + likelihood)

    data_model.set_graph(original_graph)
    return log_sum_exp(logevidence)


def get_log_evidence_meanfield(data_model: DataModelWrapper, config: Config):
    log_joint = data_model.get_log_joint()
    log_posterior = get_log_posterior_meanfield(data_model, config)
    return log_joint - log_posterior


def get_log_evidence_exact_meanfield(data_model: DataModelWrapper, config: Config):
    log_joint = data_model.get_log_joint()
    log_posterior = get_log_posterior_exact_meanfield(data_model, config)
    return log_joint - log_posterior


def get_log_evidence(data_model: DataModelWrapper, config: Config = None, **kwargs):
    config = Config(**kwargs) if config is None else config
    method = config.get("method", "meanfield")
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


def get_log_posterior_arithmetic(data_model: DataModelWrapper, config: Config):
    log_joint = data_model.get_log_joint()
    log_evidence = get_log_evidence_arithmetic(data_model, config)
    return log_joint - log_evidence


def get_log_posterior_harmonic(
    data_model: DataModelWrapper, config: Config, verbose: int = 0
):
    log_joint = data_model.get_log_joint()
    log_evidence = get_log_evidence_harmonic(data_model, config, verbose=verbose)
    return log_joint - log_evidence


def get_log_posterior_annealed(data_model: DataModelWrapper, config: Config):
    log_joint = data_model.get_log_joint()
    log_evidence = get_log_evidence_annealed(data_model, config)
    return log_joint - log_evidence


class EdgeCollector:
    def __init__(self):
        self.multiplicities = defaultdict(int)
        self.total_counts = defaultdict(int)

    def update(self, graph: core.UndirectedMultigraph):
        for edge in graph.edges():
            self.multiplicities[edge, graph.get_edge_multiplicity(*edge)] += 1
            self.total_counts[edge] += 1

    def mle(self, edge, multiplicity=1):
        return self.multiplicities[edge, multiplicity] / self.total_counts(*edge)

    def log_prob_estimate(self, graph):
        logp = 0
        for edge in graph.edges():
            logp += np.log(self.mle(edge, graph.get_edge_multiplicity(*edge)))
        return logp


def get_log_posterior_meanfield(
    data_model: DataModelWrapper,
    config: Config,
    verbose: int = 0,
    return_edgeprobs: bool = False,
):
    g = data_model.get_graph()
    collector = EdgeCollector()
    if not config.get("start_from_original", False):
        data_model.sample_prior()
    burn = config.burn_per_vertex * data_model.get_size()
    data_model_mcmc_sweep(data_model, num_steps=config.initial_burn, **config)

    for i in range(config.num_sweeps):
        t0 = time.time()
        _s = data_model_mcmc_sweep(data_model, num_steps=burn, **config)
        collector.update(data_model.get_graph())
        t1 = time.time()
        if verbose:
            print(
                f"Sweep {i}:",
                f"time={t1 - t0}",
                f"successes={_s}",
                f"failures={burn - _s}",
                f"likelihood={data_model.get_log_likelihood()}",
                f"prior={data_model.get_log_prior()}",
            )

    if config.get("reset_graph", True):
        data_model.set_graph(g)

    if return_edgeprobs:
        return {
            edge: collector.mle(edge, g.get_edge_multiplicity(edge))
            for edge in g.edges()
        }
    return collector.log_prob_estimate(g)


def get_log_posterior_annealed(
    data_model: DataModelWrapper, config: Config, verbose: int = 0
):
    log_joint = data_model.get_log_joint()
    log_evidence = get_log_evidence_annealed(data_model, config, verbose)
    return log_joint - log_evidence


def get_log_posterior_exact(data_model: DataModelWrapper, config: Config):
    log_joint = data_model.get_log_joint()
    log_evidence = get_log_evidence_exact(data_model, config)
    return log_joint - log_evidence


def get_log_posterior_exact_meanfield(data_model: DataModelWrapper, config: Config):
    original_graph = data_model.get_graph()
    size = data_model.get_size()
    graph_prior = data_model.graph_prior
    edge_count = graph_prior.get_edge_count()
    allow_self_loops = graph_prior.with_self_loops()
    allow_multiedges = graph_prior.with_parallel_edges()

    i = 0
    edge_weights = defaultdict(lambda: defaultdict(list))
    edge_total = defaultdict(list)
    evidence = get_log_evidence_exact(data_model, config)
    for g in enumerate_all_graphs(size, edge_count, allow_self_loops, allow_multiedges):
        if graph_prior.is_compatible(g):
            i += 1
            data_model.set_graph(g)
            weight = data_model.get_log_joint() - evidence
            for e, w in graphinf.utility.get_weighted_edge_list(g).items():
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


def get_log_posterior(data_model: DataModelWrapper, config: Config = None, **kwargs):
    config = Config(**kwargs) if config is None else config
    method = config.get("method", "meanfield")
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
    data_model: DataModelWrapper, config: Config, verbose: int = 0
) -> float:

    graph_prior = data_model.graph_prior
    original_graph = data_model.get_graph()
    collector = EdgeCollector()
    for _ in range(config.num_sweeps):
        graph_prior.sample()
        collector.update(graph_prior.get_state())
    return collector.log_posterior_estimate(original_graph)


def get_graph_log_evidence_meanfield(model: RandomGraphWrapper, config: Config):
    if not model.labeled:
        return model.get_log_joint()
    if importlib.util.find_spec("graph_tool"):
        from graph_tool.inference import ModeClusterState, mcmc_equilibrate
    else:
        import warnings

        warnings.warn("`graph_tool` has not been found, proceeding anyway.")
        return 0.0
    og_p = model.get_labels()
    og_g = model.get_state()
    burn = config.get("burn_per_vertex", 10) * model.get_size()

    if not config.get("start_from_original", True):
        model.sample()
        model.set_state(og_g)

    model.metropolis_sweep(burn=config.get("initial_burn", burn))

    # callback = CollectPartitionOnSweep(nested=graph_model.nested)
    # mcmc.insert_callback("partitions", callback)
    partitions = []
    for i in range(config.get("num_sweeps", 100)):
        # _s, _f = mcmc.do_MH_sweep(burn=burn)
        model.metropolis_sweep(burn=burn)
        if model.nested:
            partitions.append(model.get_nested_labels())
        else:
            partitions.append(model.get_labels())

    # partitions = callback.get_data()
    pmodes = ModeClusterState(partitions, nested=model.nested)  # from graph-tool
    if config.get("equilibrate_mode_cluster", False):
        mcmc_equilibrate(pmodes, force_niter=1, verbose=True)
    samples = []
    for p in partitions:
        model.set_labels(p)
        samples.append(model.get_log_joint() + loggamma(1 + len(np.unique(p))))

    log_evidence = np.mean(samples) + pmodes.posterior_entropy()

    model.set_labels(og_p)
    return log_evidence


def get_graph_log_evidence_exact(model: RandomGraphWrapper, config: Config) -> float:
    if model.nested:
        raise TypeError("`graph_model` must not be nested.")

    logp = []
    og_p = model.get_labels()
    for p in enumerate_all_partitions(model.get_size()):
        model.set_labels(p, False)
        logp.append(model.get_log_joint())

    log_evidence = log_sum_exp(logp)
    model.set_labels(og_p)
    return log_evidence


# def get_graph_log_evidence_annealed(
#     graph_model: RandomGraphWrapper, config: Config, verbose=False
# ) -> float:

#     mcmc = PartitionReconstructionMCMC(graph_model)
#     callback = CollectLikelihoodOnSweep()
#     mcmc.insert_callback("likelihoods", callback)
#     og_p = mcmc.get_labels()

#     burn = config.get("burn_per_vertex", 10) * graph_model.get_size()
#     logp = []

#     beta_k = np.linspace(0, 1, config.get("num_betas", 100) + 1) ** (
#         1 / config.get("exp_betas", 0.5)
#     )
#     for lb, ub in zip(beta_k[:-1], beta_k[1:]):
#         if verbose:
#             print(f"beta: {lb}")
#         mcmc.set_beta_likelihood(lb)
#         if config.get("start_from_original", True):
#             graph_model.set_labels(og_p)
#         else:
#             graph_model.sample_prior()
#         s, f = mcmc.do_MH_sweep(burn=config.get("initial_burn", burn))

#         for i in range(config.get("num_sweeps", 1000)):
#             mcmc.do_MH_sweep(burn=burn)
#         logp_k = (ub - lb) * np.array(callback.get_data())
#         logp.append(log_mean_exp(logp_k))
#         callback.clear()

#     log_evidence = sum(logp)
#     graph_model.set_labels(og_p)
#     return log_evidence


def get_graph_log_evidence(
    graph_model: RandomGraphWrapper, config: Config = None, **kwargs
) -> float:
    config = Config(**kwargs) if config is None else config

    method = config.get("method", "meanfield")

    if not graph_model.labeled:
        return graph_model.get_log_joint()
    functions = {
        "exact": get_graph_log_evidence_exact,
        "meanfield": get_graph_log_evidence_meanfield,
        # "annealed": get_graph_log_evidence_annealed,
    }
    if method in functions:
        return functions[method](graph_model, config)
    else:
        message = (
            f"Invalid method {method}, valid methods are {list(functions.keys())}."
        )
        raise ValueError(message)
