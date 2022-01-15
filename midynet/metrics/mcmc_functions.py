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


# def exact_logPosterior(model, state=None, params=None, **kwargs):
#     if params is not None:
#         model.set_params(params)
#     hx = exact_logEvidence(model, state=state, **kwargs)
#     hxp = -model.entropy(state=state)
#     return hxp - hx
#
#
# def annealed_logPosterior(model, state=None, params=None, **kwargs):
#     if params is not None:
#         model.set_params(params)
#     hx = annealed_logEvidence(model, state=state, **kwargs)
#     hxp = -model.entropy(state=state)
#     return hxp - hx
#
#
# def harmonic_logPosterior(model, state=None, params=None, **kwargs):
#     if params is not None:
#         model.set_params(params)
#     hx = harmonic_logEvidence(model, state=state, **kwargs)
#     hxp = -model.entropy(state=state)
#     return hxp - hx
#
#
# def arithmetic_logPosterior(model, state=None, params=None, **kwargs):
#     if params is not None:
#         model.set_params(params)
#     hx = arithmetic_logEvidence(model, state=state, **kwargs)
#     hxp = -model.entropy(state=state)
#     return hxp - hx
#
#
# def meanfield_logPosterior(model, state=None, params=None, **kwargs):
#     kwargs.setdefault("verbose", 0)
#     if params is not None:
#         model.set_params(params)
#     mcmc_equilibriate(model, state=state, collect=True, **kwargs)
#     hg_x = -model.graph.marginal_entropy()
#     return hg_x
#
#
# def exact_meanfield_logPosterior(model, state=None, params=None, **kwargs):
#     if params is not None:
#         model.set_params(params)
#
#     g = model.graph.copy_state()
#     edge_posterior = defaultdict(int)
#     logp = []
#     evidence = exact_logEvidence(model, state=state)
#     for gg in model.graph.all_graphs():
#         model.set_graph(gg)
#         h = -model.entropy(state=state) - evidence
#         logp.append(h)
#         for e in gg.edges():
#             i, j = sort_index(int(e.source()), int(e.target()))
#             edge_posterior[i, j] += np.exp(h)
#     model.set_graph(g)
#
#     hg_x = 0
#     for (i, j), pp in edge_posterior.items():
#         if g.edge(i, j) is not None:
#             hg_x += np.log(pp)
#         else:
#             hg_x += np.log(1 - pp)
#     return hg_x
