import multiprocessing as mp
import numpy as np
import time

from collections import defaultdict
from midynet.dynamics import get_dynamics
from midynet.graphs import get_graph, sort_index
from .statistics import MCStatistics
from .utilities import log_sum_exp
from .verbose import Verbose
from .utilities import log_mean_exp

# from .multiprocess import MultiProcess


def mcmc_equilibriate(
    model,
    state=None,
    num_sweeps=100,
    num_init_sweeps=0,
    max_num_moves=np.inf,
    init_flips=1,
    max_flips=5,
    max_consecutive_resamples=np.inf,
    adaptive=False,
    resample=0.5,
    force_resample=0.0,
    queue=None,
    collect=False,
    verbose=0,
    **kwargs,
):
    output = {}
    if state is None:
        state = model.copy_state()
    params = model.copy_params()
    model.set_state(state)
    total = 0
    consecutive_resamples = 0
    total_successes = 0
    total_failures = 0
    min_dS, max_dS = np.inf, -np.inf
    flips = init_flips
    for i in range(num_sweeps + num_init_sweeps):
        is_resampled = False
        t0 = time.time()

        success, failure, dS1, dS2 = model.mcmc_sweep(flips=flips, **kwargs)

        min_dS = dS1 if min_dS > dS1 else min_dS
        max_dS = dS2 if max_dS < dS2 else max_dS
        if consecutive_resamples > max_consecutive_resamples:
            flips = 1
        if i > num_init_sweeps:
            if collect:
                model.collect()
            if queue is not None:
                queue.append(model.copy_params())
        if adaptive:
            if success > failure and flips < max_flips:
                flips += 1
            elif failure > success and flips > 1:
                flips -= 1
        if flips == max_flips:
            if np.random.rand() < resample:
                model.resample()
                is_resampled = True
            consecutive_resamples += 1
        if np.random.rand() < force_resample:
            model.resample()
            is_resampled = True

        else:
            consecutive_resamples = 0
        total_successes += success * flips
        total_failures += failure * flips
        if (
            total_successes > max_num_moves
            or max_consecutive_resamples < consecutive_resamples
        ):
            break
        t1 = time.time()
        if verbose == 1:
            print(
                f"Sweeps: {i}\t "
                + f"Total moves: {total_successes}\t "
                + f"Consecutive: {success * flips}\t "
                + f"Failure: {failure * flips}\t "
                + f"Multiflip: {flips}\t "
                + f"min(dS): {np.round(min_dS, 4)}\t "
                + f"max(dS): {np.round(max_dS, 4)}\t "
                + f"Resampled: {is_resampled}\t "
                + f"time: {t1 - t0}"
            )

    model.set_state(state)
    model.set_params(params)
    return total_successes, total_failures


def logEvidence(model, state=None, method=None, **kwargs):
    method = "meanfield" if method is None else method
    if method == "exact":
        logF = exact_logEvidence(model, state=state, **kwargs)
    elif method == "annealed":
        logF = annealed_logEvidence(model, state=state, **kwargs)
    elif method == "harmonic":
        logF = harmonic_logEvidence(model, state=state, **kwargs)
    elif method == "arithmetic":
        logF = arithmetic_logEvidence(model, state=state, **kwargs)
    elif method == "meanfield":
        logF = meanfield_logEvidence(model, state=state, **kwargs)
    elif method == "exact_meanfield":
        logF = exact_meanfield_logEvidence(model, state=state, **kwargs)

    else:
        l = [
            "exact",
            "annealed",
            "harmonic",
            "arithmetic",
            "meanfield",
            "exact_meanfield",
        ]
        msg = f"Invalid method {method}, must be {l}."
        raise RuntimeError(msg)
    return logF


def exact_logEvidence(model, state=None, **kwargs):
    g = model.graph.copy_state()
    logp = []
    for gg in model.graph.all_graphs():
        model.set_graph(gg)
        logp.append(-model.entropy(state=state))
    model.set_graph(g)
    return log_sum_exp(logp)


def annealed_logEvidence(model, state=None, kmax=10, alpha=0.5, **kwargs):
    kwargs.setdefault("verbose", 0)
    logF = []
    beta_k = (np.linspace(0, 1, kmax + 1)) ** (1.0 / alpha)
    lower_betas = beta_k[:-1]
    upper_betas = beta_k[1:]
    params = model.copy_params()
    for lb, ub in zip(lower_betas, upper_betas):
        logr_k = []
        queue = []
        if kwargs["verbose"] == 1 or kwargs["verbose"] == 2:
            print(f"Inverse temperature: {lb}")
        mcmc_equilibriate(model, state=state, queue=queue, beta=lb, **kwargs)
        ll = []
        for p in queue:
            model.set_params(p)
            logr_k.append((ub - lb) * model.loglikelihood(state=state))
        logF.append(log_mean_exp(logr_k))
    model.set_params(params)
    return np.sum(logF)


def arithmetic_logEvidence(model, state=None, num_sweeps=10, M=10, **kwargs):
    logF = []
    params = model.copy_params()
    for m in range(M):
        logF_k = []
        for k in range(num_sweeps):
            model.resample()
            logF_k.append(model.loglikelihood(state))
        logF.append(log_mean_exp(logF_k))
    model.set_params(params)
    return np.mean(logF)


def harmonic_logEvidence(model, state=None, **kwargs):
    queue = []
    kwargs.setdefault("verbose", 0)
    params = model.copy_params()
    mcmc_equilibriate(model, state=state, queue=queue, **kwargs)
    logF = []
    for p in queue:
        model.set_params(p)
        logF.append(-model.loglikelihood(state))
    model.set_params(params)
    return -log_mean_exp(logF)


def meanfield_logEvidence(model, state=None, **kwargs):
    kwargs.setdefault("verbose", 0)
    hp_x = meanfield_logPosterior(model, state=state, **kwargs)
    hxp = -model.entropy()
    return hxp - hp_x


def exact_meanfield_logEvidence(model, state=None, **kwargs):
    kwargs.setdefault("verbose", 0)
    hp_x = exact_meanfield_logPosterior(model, state=state, **kwargs)
    hxp = -model.entropy()
    return hxp - hp_x


def logPosterior(model, state=None, params=None, method=None, **kwargs):
    method = "meanfield" if method is None else method
    if method == "exact":
        logR = exact_logPosterior(model, state=state, params=params, **kwargs)
    elif method == "annealed":
        logR = annealed_logPosterior(model, state=state, params=params, **kwargs)
    elif method == "harmonic":
        logR = harmonic_logPosterior(model, state=state, params=params, **kwargs)
    elif method == "arithmetic":
        logR = arithmetic_logPosterior(model, state=state, params=params, **kwargs)
    elif method == "meanfield":
        logR = meanfield_logPosterior(model, state=state, params=params, **kwargs)
    elif method == "exact-meanfield":
        logR = exact_meanfield_logPosterior(model, state=state, params=params, **kwargs)
    else:
        l = [
            "exact",
            "annealed",
            "harmonic",
            "arithmetic",
            "meanfield",
            "exact-meanfield",
        ]
        msg = f"Invalid method {method}, must be {l}."
        raise RuntimeError(msg)
    return logR


def exact_logPosterior(model, state=None, params=None, **kwargs):
    if params is not None:
        model.set_params(params)
    hx = exact_logEvidence(model, state=state, **kwargs)
    hxp = -model.entropy(state=state)
    return hxp - hx


def annealed_logPosterior(model, state=None, params=None, **kwargs):
    if params is not None:
        model.set_params(params)
    hx = annealed_logEvidence(model, state=state, **kwargs)
    hxp = -model.entropy(state=state)
    return hxp - hx


def harmonic_logPosterior(model, state=None, params=None, **kwargs):
    if params is not None:
        model.set_params(params)
    hx = harmonic_logEvidence(model, state=state, **kwargs)
    hxp = -model.entropy(state=state)
    return hxp - hx


def arithmetic_logPosterior(model, state=None, params=None, **kwargs):
    if params is not None:
        model.set_params(params)
    hx = arithmetic_logEvidence(model, state=state, **kwargs)
    hxp = -model.entropy(state=state)
    return hxp - hx


def meanfield_logPosterior(model, state=None, params=None, **kwargs):
    kwargs.setdefault("verbose", 0)
    if params is not None:
        model.set_params(params)
    mcmc_equilibriate(model, state=state, collect=True, **kwargs)
    hg_x = -model.graph.marginal_entropy()
    return hg_x


def exact_meanfield_logPosterior(model, state=None, params=None, **kwargs):
    if params is not None:
        model.set_params(params)

    g = model.graph.copy_state()
    edge_posterior = defaultdict(int)
    logp = []
    evidence = exact_logEvidence(model, state=state)
    for gg in model.graph.all_graphs():
        model.set_graph(gg)
        h = -model.entropy(state=state) - evidence
        logp.append(h)
        for e in gg.edges():
            i, j = sort_index(int(e.source()), int(e.target()))
            edge_posterior[i, j] += np.exp(h)
    model.set_graph(g)

    hg_x = 0
    for (i, j), pp in edge_posterior.items():
        if g.edge(i, j) is not None:
            hg_x += np.log(pp)
        else:
            hg_x += np.log(1 - pp)
    return hg_x
