from midynet.config import *
from _midynet.mcmc import DynamicsMCMC
from _midynet.mcmc.callbacks import CollectLikelihoodOnSweep
from _midynet.random_graph import RandomGraph
from _midynet.dynamics import Dynamics


def log_evidence_arithmetic(dynamicsMCMC: DynamicsMCMC, config: Config):
    logp = []
    for k in range(config.K):
        logp_k = []
        for k in range(config.num_sweeps):
            dynamicsMCMC.sample_graph()
            logp_k.append(dynamicsMCMC.get_log_likelihood())
        logp.append(log_mean_exp(logp_k))
    return np.mean(logp)


def log_evidence_harmonic(dynamicsMCMC: DynamicsMCMC, config: Config):
    callback = CollectLikelihoodOnSweep()
    dynamicsMCMC.add_callback(callback)
    dynamicsMCMC.set_up()

    for i in range(config.num_sweeps):
        dynamicsMCMC.do_MH_sweep(burn=config.burn)
    logp = -np.array(callback.get_log_likelihoods())

    dynamicsMCMC.tear_down()
    dynamicsMCMC.pop_callback()

    return -log_mean_exp(logp)


def log_evidence_meanfield(dynamicsMCMC: DynamicsMCMC, config: Config):
    return dynamicsMCMC.get_log_joint() - log_posterior_meanfield(dynamicsMCMC, config)


def log_evidence_annealed(dynamicsMCMC: DynamicsMCMC, config: Config):
    callback = CollectLikelihoodOnSweep()
    g = dynamicsMCMC.get_graph()
    dynamicsMCMC.add_callback(callback)
    dynamicsMCMC.set_up()

    logp = []
    for lb, ub in zip(config.beta_k[:-1], config.beta_k[1:]):
        dynamicsMCMC.set_beta_likelihood(lb)
        if config.reset_to_original:
            dynamicsMCMC.set_graph(g)
        for i in range(config.num_sweeps):
            dynamicsMCMC.do_MH_sweep(burn=config.burn)
        logp_k = (ub - lb) * np.array(callback.get_log_likelihoods())
        logp.append(log_mean_exp(logp_k))
        callback.clear()

    dynamicsMCMC.tear_down()
    dynamicsMCMC.pop_callback()
    return -log_mean_exp(logp)


def log_evidence_exact(dynamicsMCMC: DynamicsMCMC, config: Config):
    raise NotImplementedError()


def log_evidence_exact_meanfield(dynamicsMCMC: DynamicsMCMC, config: Config):
    return dynamicsMCMC.get_log_joint() - log_posterior_exact_meanfield(
        dynamicsMCMC, config
    )


def log_evidence(dynamicsMCMC: DynamicsMCMC, config: Config):
    method = "meanfield" if "method" not in config else config.method
    if method == "exact":
        return log_evidence_exact(dynamicsMCMC, config)
    elif method == "exact_meanfield":
        return log_evidence_exact_meanfield(dynamicsMCMC, config)
    elif method == "arithmetic":
        return log_evidence_arithmetic(dynamicsMCMC, config)
    elif method == "harmonic":
        return log_evidence_harmonic(dynamicsMCMC, config)
    elif method == "meanfield":
        return log_evidence_meanfield(dynamicsMCMC, config)
    elif method == "annealed":
        return log_evidence_annealed(dynamicsMCMC, config)


def log_posterior_arithmetic(dynamicsMCMC: DynamicsMCMC, config: Config):
    return dynamicsMCMC.get_log_joint() - log_evidence_arithmetic(dynamicsMCMC, config)


def log_posterior_harmonic(dynamicsMCMC: DynamicsMCMC, config: Config):
    return dynamicsMCMC.get_log_joint() - log_evidence_harmonic(dynamicsMCMC, config)


def log_posterior_meanfield(dynamicsMCMC: DynamicsMCMC, config: Config):
    callback = CollectEdgeMultiplicityOnSweep()
    dynamicsMCMC.add_callback(callback)
    dynamicsMCMC.set_up()

    for i in range(config.num_sweeps):
        dynamicsMCMC.do_MH_sweep(burn=config.burn)
    h = callback.get_marginal_entropy()

    dynamicsMCMC.tear_down()
    dynamicsMCMC.pop_callback()

    return h


def log_posterior_annealed(dynamicsMCMC: DynamicsMCMC, config: Config):
    return dynamicsMCMC.get_log_joint() - log_evidence_annealed(dynamicsMCMC, config)


def log_posterior_exact(dynamicsMCMC: DynamicsMCMC, config: Config):
    return dynamicsMCMC.get_log_joint() - log_evidence_exact(dynamicsMCMC, config)


def log_posterior_exact_meanfield(dynamicsMCMC: DynamicsMCMC, config: Config):
    raise NotImplementedError()


def log_posterior(dynamicsMCMC: DynamicsMCMC, config: Config):
    method = "meanfield" if "method" not in config else config.method
    if method == "exact":
        return log_posterior_exact(dynamicsMCMC, config)
    elif method == "exact_meanfield":
        return log_posterior_exact_meanfield(dynamicsMCMC, config)
    elif method == "arithmetic":
        return log_posterior_arithmetic(dynamicsMCMC, config)
    elif method == "harmonic":
        return log_posterior_harmonic(dynamicsMCMC, config)
    elif method == "meanfield":
        return log_posterior_meanfield(dynamicsMCMC, config)
    elif method == "annealed":
        return log_posterior_annealed(dynamicsMCMC, config)
