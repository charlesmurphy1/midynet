import midynet
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt

from midynet.util.convert import (
    convert_basegraph_to_graphtool,
    convert_basegraph_to_networkx,
    convert_gt_blockstate_to_partition,
)
from midynet.util.display import draw_graph


def main():
    N = 100
    E = 1000
    B = 3
    c = midynet.config.RandomGraphConfig.planted_partition(
        size=N,
        edge_count=E,
        block_count=B,
        assortativity=0.4,
        stub_labeled=True,
    )
    test_model = midynet.config.RandomGraphFactory.build(c)

    prior = midynet.random_graph.StochasticBlockModelFamily(
        N,
        E,
        B,
        block_hyperprior=True,
        canonical=False,
        stub_labeled=True,
        block_proposer_type="mixed",
    )
    prior.set_state(test_model.get_state())
    gt_graph = convert_basegraph_to_graphtool(test_model.get_state())
    init_labels = convert_gt_blockstate_to_partition(
        gt.minimize_blockmodel_dl(gt_graph)
    )
    prior.set_labels(init_labels)
    pos = nx.spring_layout(convert_basegraph_to_networkx(prior.get_state()))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    draw_graph(
        test_model.get_state(),
        ax=ax[0],
        labels=test_model.get_labels(),
        pos=pos,
        with_self_loops=False,
        with_parallel_edges=False,
    )
    draw_graph(
        test_model.get_state(),
        ax=ax[1],
        labels=prior.get_labels(),
        pos=pos,
        with_self_loops=False,
        with_parallel_edges=False,
    )
    mcmc = midynet.config.PartitionMCMC(prior)
    callback = midynet.mcmc.callbacks.CollectPartitionOnSweepForCommunity()
    mcmc.insert_callback("partition", callback)
    for i in range(100):
        mcmc.do_MH_sweep(10 * N)
    pmode = gt.PartitionModeState(callback.get_data(), converge=True)
    cluster = gt.ModeClusterState(callback.get_data(), B=1)
    gt.mcmc_equilibrate(cluster, force_niter=100, verbose=True)
    pv = pmode.get_marginal(gt_graph)
    #
    marginals = np.array(pv.get_2d_array(gt_graph.get_vertices())).astype("float").T
    marginals /= marginals.sum(-1)
    max_label = np.max(np.array(callback.get_data()))
    marginals = marginals[:, : max_label + 1]
    draw_graph(
        mcmc.get_graph(),
        mcmc.get_labels(),
        ax=ax[2],
        pos=pos,
        marginals=marginals,
        with_self_loops=False,
        with_parallel_edges=False,
    )
    posterior_entropy = 0
    for p in callback.get_data():
        rp = pmode.relabel_partition(p)
        logP = pmode.posterior_lprob(rp, MLE=True)
        posterior_entropy -= np.exp(logP) * logP

    mf_entropy = 0
    for m in marginals:
        m = m[m > 0]
        mf_entropy -= np.sum(m * np.log(m))
    print(
        posterior_entropy,
        mf_entropy,
        pmode.posterior_entropy(True),
        pmode.posterior_entropy(False),
        cluster.posterior_entropy(True),
    )
    print(cluster.get_wr())
    plt.show()


if __name__ == "__main__":
    main()
