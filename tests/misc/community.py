import midynet
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt

from midynet.util.convert import get_graphtool_graph_from_basegraph
from midynet.util.display import draw_graph


def main():
    config = midynet.config.ExperimentConfig.community("test", "hyperuniform_sbm")
    size = 100
    E = 1000
    config.graph.set_value("size", size)
    config.graph.blocks.set_value("size", size)
    config.graph.blocks.block_count.set_value("max", 5)
    config.graph.block_proposer.set_value("sample_label_count_prob", 0.5)
    config.graph.edge_matrix.edge_count.set_value("state", E)
    print(config.format())
    c2 = midynet.config.RandomGraphConfig.planted_partition(
        size=size, edge_count=E, block_count=3, assortativity=0.8
    )
    test_graph = midynet.config.RandomGraphFactory.build(c2)
    test_graph.sample()
    true_labels = test_graph.get_labels()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    pos = nx.spring_layout(get_networkx_graph_from_basegraph(test_graph.get_graph()))
    draw_graph(test_graph.get_graph(), test_graph.get_labels(), ax=ax[0], pos=pos)
    ax[0].set_title("True partition")
    ax[0].axis("off")

    mcmc = midynet.config.MCMCFactory.build_community(config)
    likelihood = midynet.mcmc.callbacks.CollectLikelihoodOnSweep()
    prior = midynet.mcmc.callbacks.CollectPriorOnSweep()
    partition = midynet.mcmc.callbacks.CollectPartitionOnSweepForCommunity()
    mcmc.insert_callback("likelihood", likelihood)
    mcmc.insert_callback("prior", prior)
    mcmc.insert_callback("partition", partition)
    mcmc.sample()
    mcmc.set_graph(test_graph.get_graph())
    mcmc.set_labels(true_labels)
    mcmc.set_up()

    mcmc.set_beta_likelihood(0.0)
    mcmc.set_beta_prior(0.1)
    print(f"Initial randomization")
    mcmc.do_MH_sweep(50_000)
    draw_graph(mcmc.get_graph(), mcmc.get_labels(), ax=ax[1], pos=pos)
    ax[1].set_title("Initial guess")
    ax[1].axis("off")
    mcmc.set_beta_likelihood(1)
    mcmc.set_beta_prior(1)
    for i in range(100):
        print(f"Sweep {i}")
        mcmc.do_MH_sweep(50_000)

    pmode = gt.PartitionModeState(partition.get_data(), converge=True)
    gt_graph = get_graphtool_graph_from_basegraph(test_graph.get_graph())
    pv = pmode.get_marginal(gt_graph)

    marginals = np.array(pv.get_2d_array(gt_graph.get_vertices())).astype("float").T
    marginals /= marginals.sum(-1)
    max_label = np.max(np.array(partition.get_data()))
    marginals = marginals[:, : max_label + 1]
    draw_graph(
        mcmc.get_graph(), mcmc.get_labels(), ax=ax[2], pos=pos, marginals=marginals
    )
    ax[2].set_title("Inferred partition")
    ax[2].axis("off")
    plt.show()


if __name__ == "__main__":
    main()
