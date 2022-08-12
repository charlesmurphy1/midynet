import midynet
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt

from midynet.util.convert import (
    convert_basegraph_to_graphtool,
    convert_basegraph_to_networkx,
)
from midynet.util.display import draw_graph


def main():
    N = 100
    E = 250
    B = 3
    c = midynet.config.RandomGraphConfig.planted_partition(
        sizes=[N / 3, N / 3, N / 3], edge_count=E, assortativity=0.8
    )
    test_graph = midynet.config.RandomGraphFactory.build(c)
    graph_prior = midynet.random_graph.StochasticBlockModelFamily(
        N, E, B, block_hyperprior=True, canonical=False, stub_labeled=False
    )
    graph_prior.set_state(test_graph.get_state())
    pos = nx.spring_layout(convert_basegraph_to_networkx(graph_prior.get_state()))

    mcmc = midynet.config.PartitionMCMC(graph_prior)
    draw_graph(mcmc.get_graph(), test_graph.get_labels(), pos=pos)
    plt.show()
    callback = midynet.mcmc.callbacks.CollectPartitionOnSweepForCommunity()
    mcmc.insert_callback("partition", callback)
    for i in range(100):
        mcmc.do_MH_sweep(100 * N)
    pmode = gt.PartitionModeState(callback.get_data())
    gt_graph = convert_basegraph_to_graphtool(graph_prior.get_state())
    pv = pmode.get_marginal(gt_graph)

    marginals = np.array(pv.get_2d_array(gt_graph.get_vertices())).astype("float").T
    marginals /= marginals.sum(-1)
    max_label = np.max(np.array(callback.get_data()))
    marginals = marginals[:, : max_label + 1]
    draw_graph(mcmc.get_graph(), mcmc.get_labels(), pos=pos, marginals=marginals)
    plt.show()
    posterior_entropy = 0
    for p in callback.get_data():
        rp = pmode.relabel_partition(p)
        logP = pmode.posterior_lprob(relabeled_p, MLE=True)
        posterior_entropy -= np.exp(logP) * logP


if __name__ == "__main__":
    main()
