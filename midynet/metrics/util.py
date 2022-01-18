import importlib
import numpy as np

from midynet.config import *
from midynet.util import log_mean_exp
from _midynet.mcmc import DynamicsMCMC, RandomGraphMCMC
from _midynet.mcmc.callbacks import (
    CollectLikelihoodOnSweep,
    CollectEdgeMultiplicityOnSweep,
    CollectPartitionOnSweep,
)


__all__ = ["get_log_evidence", "get_log_posterior"]


def get_log_evidence_arithmetic(dynamicsMCMC: DynamicsMCMC, config: Config):
    logp = []
    g = dynamicsMCMC.get_graph()
    for k in range(config.K):
        logp_k = []
        for m in range(config.num_sweeps):
            dynamicsMCMC.sample_graph()
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
    for i in range(config.num_sweeps):
        dynamicsMCMC.do_MH_sweep(burn=burn)
    logp = -np.array(callback.get_log_likelihoods())

    dynamicsMCMC.tear_down()
    dynamicsMCMC.pop_callback()
    dynamicsMCMC.set_graph(g)
    return -log_mean_exp(logp)


def get_log_evidence_meanfield(dynamicsMCMC: DynamicsMCMC, config: Config):
    return dynamicsMCMC.get_log_joint() - get_log_posterior_meanfield(
        dynamicsMCMC, config
    )


def get_log_evidence_annealed(dynamicsMCMC: DynamicsMCMC, config: Config):
    callback = CollectLikelihoodOnSweep()
    g = dynamicsMCMC.get_graph()
    dynamicsMCMC.add_callback(callback)
    dynamicsMCMC.set_up()
    burn = config.burn_per_vertex * dynamicsMCMC.get_dynamics().get_size()
    logp = []
    for lb, ub in zip(config.beta_k[:-1], config.beta_k[1:]):
        dynamicsMCMC.set_beta_likelihood(lb)
        if config.reset_to_original:
            dynamicsMCMC.set_graph(g)
        for i in range(config.num_sweeps):
            dynamicsMCMC.do_MH_sweep(burn=burn)
        logp_k = (ub - lb) * np.array(callback.get_log_likelihoods())
        logp.append(log_mean_exp(logp_k))
        callback.clear()

    dynamicsMCMC.tear_down()
    dynamicsMCMC.pop_callback()
    dynamicsMCMC.set_graph(g)

    return -log_mean_exp(logp)


def get_log_evidence_exact(dynamicsMCMC: DynamicsMCMC, config: Config):
    raise NotImplementedError()


def get_log_evidence_exact_meanfield(dynamicsMCMC: DynamicsMCMC, config: Config):
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
            f"Invalid method {method}, valid methods are {list(functions.keys())}."
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
    dynamicsMCMC.add_callback(graph_callback)
    dynamicsMCMC.set_up()
    burn = config.burn_per_vertex * dynamicsMCMC.get_dynamics().get_size()
    for i in range(config.num_sweeps):
        dynamicsMCMC.do_MH_sweep(burn=burn)
    logp = -graph_callback.get_marginal_entropy()  # -H(G|X)

    dynamicsMCMC.tear_down()
    dynamicsMCMC.pop_callback()

    return logp


def get_log_posterior_meanfield_sbm(dynamicsMCMC: DynamicsMCMC, config: Config):
    if importlib.util.find_spec("graph_tool") is None:
        message = (
            f"The meanfield method cannot be used for SBM graphs, "
            + "because `graph_tool` is not installed."
        )
        raise NotImplementedError(message)
    else:
        from graph_tool.inference import (
            PartitionModeState,
            ModeClusterState,
            mcmc_equilibriate,
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
        mcmc_equilibrate(partition_modes, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
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
    return dynamicsMCMC.get_log_joint() - get_log_evidence_exact(dynamicsMCMC, config)


def get_log_posterior_exact_meanfield(dynamicsMCMC: DynamicsMCMC, config: Config):
    raise NotImplementedError()


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
            f"Invalid method {method}, valid methods are {list(functions.keys())}."
        )
        raise ValueError(message)
