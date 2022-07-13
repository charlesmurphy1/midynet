import midynet
import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt
import networkx as nx

from scipy.special import loggamma
from midynet.util import display, convert
from collections import defaultdict


def fact(k):
    return int(np.exp(loggamma(k + 1)))


def logfact(k):
    return loggamma(k + 1)


def binom(n, k):
    return int(fact(n) / fact(n - k) / fact(k))


def logbase(x, base=np.e):
    return np.log(x) / np.log(base)


def to_nary(x, base=2, dim=None):
    if type(x) is int or type(x) is float:
        x = np.array([x])
    if dim is None:
        max_val = base ** np.floor(logbase(np.max(x), base) + 1)
        dim = int(logbase(max_val, base))
    y = np.zeros([dim, *x.shape])
    for idx, xx in np.ndenumerate(x):
        r = np.zeros(dim)
        r0 = xx
        while r0 > 0:
            b = int(np.floor(logbase(r0, base)))
            r[b] += 1
            r0 -= base**b
        y.T[idx] = r[::-1]
    return y


def enumerate_partitions(B, N):
    for i in range(B**N):
        yield tuple(to_nary(i, B, dim=N).squeeze().astype("int").tolist())


def main():
    N, E = 3, 3

    config = midynet.config.ExperimentConfig.community(
        "test",
        "uniform_sbm",
        graph_params=dict(size=N, edge_count=E),
    )
    mcmc = midynet.config.MCMCFactory.build_community(config)
    test_graph = midynet.config.RandomGraphFactory.build(
        midynet.config.RandomGraphConfig.planted_partition(N, E, 2, assortativity=0.8)
    )
    graph = mcmc.graph
    test_graph.sample()

    mcmc.sample()
    mcmc.set_graph(test_graph.get_graph())
    mcmc.set_labels(test_graph.get_labels())
    mcmc.set_up()

    logPG = []
    og_p = mcmc.get_labels().copy()
    og_index = 0
    all_partitions = list(enumerate_partitions(N, N))
    for i, p in enumerate(all_partitions):
        mcmc.set_labels(p)
        logPG.append(mcmc.get_log_joint())

        if p == tuple(og_p):
            og_index = i
    logPG = midynet.util.log_sum_exp(logPG)

    callback = midynet.mcmc.callbacks.CollectPartitionOnSweepForCommunity()
    mcmc.insert_callback("partitions", callback)
    mcmc.set_up()
    mcmc.do_MH_sweep(1000)
    for i in range(50000):
        mcmc.do_MH_sweep(500)
    gt_graph = convert.get_graphtool_graph_from_basegraph(mcmc.get_graph())
    pmode = gt.PartitionModeState(callback.get_data(), converge=True)

    posterior = {"true": {}, "estimated": {}, "mf": {}}
    aggregate = {"true": {}, "estimated": {}, "mf": {}}

    # relabeled_og_p = tuple(pmode.relabel_partition(og_p).astype("int").tolist())
    # relabeled_og_index = 0
    true_agg, estimated_agg, mf_agg = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    counts = defaultdict(int)
    logz = []
    all_relabeled_partitions = []
    for p in callback.get_data():
        counts[tuple(p)] += 1
    for i, p in enumerate(all_partitions):
        mcmc.set_labels(p)
        c = counts[p] if p in counts else 0
        relabeled_p = pmode.relabel_partition(p)
        logP = pmode.posterior_lprob(relabeled_p, MLE=True)
        rp = tuple(relabeled_p.astype("int").tolist())
        true_agg[rp].append(mcmc.get_log_joint() - logPG)
        estimated_agg[rp].append(np.log(c / len(callback.get_data())))
        if rp not in all_relabeled_partitions:
            all_relabeled_partitions.append(rp)
        # True
        posterior["true"][p] = mcmc.get_log_joint() - logPG

        # Estimated
        posterior["estimated"][p] = np.log(c / len(callback.get_data()))

        # MF
        posterior["mf"][p] = logP - logfact(N) + logfact(N - len(np.unique(p)))
        # logz.append(logP)

    # posterior["mf"] = {
    #     k: v - midynet.util.log_sum_exp(logz) for k, v in posterior["mf"].items()
    # }

    relabeled_og_p = tuple(pmode.relabel_partition(og_p).astype("int").tolist())
    for i, p in enumerate(all_relabeled_partitions):
        aggregate["true"][p] = midynet.util.log_sum_exp(true_agg[p])
        aggregate["estimated"][p] = midynet.util.log_sum_exp(estimated_agg[p])
        aggregate["mf"][p] = pmode.posterior_lprob(p, MLE=True)
        if p == relabeled_og_p:
            relabeled_og_index = i

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    indices = range(len(all_partitions))
    ax[0].plot(indices, list(posterior["true"].values()), label="Exact")
    ax[0].plot(indices, list(posterior["estimated"].values()), label="Estimated")
    ax[0].plot(indices, list(posterior["mf"].values()), label="MF")
    ax[0].axvline(
        og_index,
        linestyle="--",
        color=display.dark_colors["grey"],
        linewidth=2,
        label="Ground truth",
    )
    ax[0].set_xticks(range(len(all_partitions)))
    ax[0].set_xticklabels([str(p) for p in all_partitions], rotation=75)
    ax[0].set_xlabel("Partitions", fontsize=12)
    ax[0].set_ylabel("Posterior probability $\log P(b|G)$", fontsize=12)
    ax[0].tick_params(labelsize=12, axis="both")

    indices = range(len(all_relabeled_partitions))
    ax[1].plot(indices, list(aggregate["true"].values()), label="Exact")
    ax[1].plot(indices, list(aggregate["estimated"].values()), label="Estimated")
    ax[1].plot(indices, list(aggregate["mf"].values()), label="MF")
    ax[1].axvline(
        relabeled_og_index,
        linestyle="--",
        color=display.dark_colors["grey"],
        linewidth=2,
        label="Ground truth",
    )
    ax[1].set_xticks(indices)
    ax[1].set_xticklabels([str(p) for p in all_relabeled_partitions], rotation=75)
    ax[1].set_xlabel("Relabeled partitions", fontsize=12)
    ax[1].set_ylabel("Posterior aggregate", fontsize=12)
    ax[1].tick_params(labelsize=12, axis="both")

    ax[0].legend(fontsize=10)
    ax[1].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig("partition-mf.png")
    plt.show()


import seaborn as sb


def mf_entropy(mcmc):
    gt_graph = convert.get_graphtool_graph_from_basegraph(mcmc.get_graph())
    pmode = gt.PartitionModeState(
        mcmc.get_label_callback("partition").get_data(), converge=True
    )
    pv = pmode.get_marginal(gt_graph)
    marginals = pv.get_2d_array(gt_graph.vertices())
    print(marginals)


def main2():
    N, E = 4, 8

    config = midynet.config.ExperimentConfig.community(
        "test",
        "uniform_sbm",
        graph_params=dict(size=N, edge_count=E),
    )
    mcmc = midynet.config.MCMCFactory.build_community(config)
    mcmc.sample()

    exact_samples = []
    mf_samples = []
    for i in range(10):
        test_graph = midynet.config.RandomGraphFactory.build(
            midynet.config.RandomGraphConfig.planted_partition(
                N, E, 2, assortativity=0.8
            )
        )
        test_graph.sample()
        mcmc.set_graph(test_graph.get_graph())
        mcmc.set_labels(test_graph.get_labels())
        callback = midynet.mcmc.callbacks.CollectPartitionOnSweepForCommunity()
        mcmc.insert_callback("partition", callback)
        mcmc.set_up()
        # og_p = test_graph.get_labels()
        logPG = []
        for i, p in enumerate(enumerate_partitions(N, N)):
            mcmc.set_labels(p)
            logPG.append(mcmc.get_log_joint())
        logPG = midynet.util.log_sum_exp(logPG)
        # mcmc.set_labels(og_p)
        exact = []
        mcmc.samplePrior()
        for p in enumerate_partitions(N, N):
            mcmc.set_labels(p)
            exact.append(np.exp(mcmc.get_log_joint() - logPG))
        exact = np.array(exact)
        exact_samples.append(np.sum(exact * np.log(exact)))

        for i in range(100):
            mcmc.do_MH_sweep(100)
        mf_entropy(mcmc)
        mcmc.remove_callback("partition")

    sb.displot(exact_samples, kde=False)
    plt.show()


if __name__ == "__main__":
    main2()
