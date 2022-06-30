import midynet
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# from graph_tool.inference import mutual_information, PartitionModeState
import graph_tool.all as gt


def main():
    config = midynet.config.ExperimentConfig.community("test", "hyperuniform_sbm")
    config.graph.blocks.block_count.set_value("max", 5)
    config.graph.block_proposer.set_value("sample_label_count_prob", 0.5)
    config.graph.edge_matrix.edge_count.set_value("state", 1000)

    c2 = midynet.config.RandomGraphConfig.planted_partition(
        size=100, edge_count=1000, block_count=3, assortativity=0.8
    )
    test_graph = midynet.config.RandomGraphFactory.build(c2)
    test_graph.sample()
    true_labels = test_graph.get_labels()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    pos = nx.spring_layout(get_networkx_graph(test_graph.get_graph()))
    draw(test_graph.get_graph(), test_graph.get_labels(), ax=ax[0], pos=pos)
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
    mcmc.do_MH_sweep(10_000)
    draw(mcmc.get_graph(), mcmc.get_labels(), ax=ax[1], pos=pos)
    ax[1].set_title("Initial guess")
    ax[1].axis("off")
    mcmc.set_beta_likelihood(1)
    mcmc.set_beta_prior(1)
    for i in range(100):
        mcmc.do_MH_sweep(10_000)

    pmode = gt.PartitionModeState(partition.get_data(), converge=True)
    gt_graph = get_graphtool_graph(test_graph.get_graph())
    pv = pmode.get_marginal(gt_graph)

    marginals = np.array(pv.get_2d_array(gt_graph.get_vertices())).astype("float").T
    marginals /= marginals.sum(-1)
    max_label = np.max(np.array(partition.get_data()))
    marginals = marginals[:, : max_label + 1]
    draw(mcmc.get_graph(), mcmc.get_labels(), ax=ax[2], pos=pos, marginals=marginals)
    ax[2].set_title("Inferred partition")
    ax[2].axis("off")
    plt.show()


from midynet.util import display


def get_edgelist(bs_graph):
    el = []
    for v in bs_graph:
        for u in bs_graph.get_out_edges_of_idx(v):
            if v > u.vertex_index:
                continue
            el.append((v, u.vertex_index))
    return el


def get_networkx_graph(bs_graph):
    nx_graph = nx.Graph()
    for v in bs_graph:
        nx_graph.add_node(v)
        for u in bs_graph.get_out_edges_of_idx(v):
            if v > u.vertex_index:
                continue
            nx_graph.add_edge(v, u.vertex_index)
    return nx_graph


def get_graphtool_graph(bs_graph):
    gt_graph = gt.Graph()
    for v in bs_graph:
        for u in bs_graph.get_out_edges_of_idx(v):
            if v > u.vertex_index:
                continue
            gt_graph.add_edge(v, u.vertex_index)
    return gt_graph


def drawPieMarker(xs, ys, ratios, colors, size=60, ax=None):
    assert sum(ratios) <= 1, "sum of ratios needs to be < 1"

    markers = []
    previous = 0
    # calculate the points of the pie pieces
    for color, ratio in zip(colors, ratios):
        this = 2 * np.pi * ratio + previous
        x = [0] + np.cos(np.linspace(previous, this, 10)).tolist() + [0]
        y = [0] + np.sin(np.linspace(previous, this, 10)).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        markers.append(
            {
                "marker": xy,
                "s": np.abs(xy).max() ** 2 * np.array(size),
                "facecolor": color,
            }
        )

    # scatter each of the pie pieces to create pies
    for marker in markers:
        ax.scatter(xs, ys, zorder=10, **marker)


def draw(graph, labels, ax=None, pos=None, marginals=None):
    ax = plt.gca() if ax is None else ax
    el = []
    node_color = [None for i in range(graph.get_size())]
    # node_color = [colors[l] for l in labels]
    colors = (
        [c for c in display.light_colors.values()]
        + [c for c in display.med_colors.values()]
        + [c for c in display.dark_colors.values()]
    )
    if pos is None:
        nx_graph = get_networkx_graph(graph)
        pos = nx.spring_layout(nx_graph)

    for i, j in get_edgelist(graph):
        pos_i, pos_j = pos[i], pos[j]
        ax.plot(
            [pos_i[0], pos_j[0]],
            [pos_i[1], pos_j[1]],
            linestyle="-",
            marker="None",
            color=display.med_colors["grey"],
            linewidth=1,
        )

    for i, p in pos.items():
        if marginals is None:
            ax.plot([p[0]], [p[1]], color=colors[labels[i]], marker="o")
        else:
            drawPieMarker([p[0]], [p[1]], marginals[i], colors=colors, ax=ax)

    # plt.show()
    # print(pos)

    # plt.show()


if __name__ == "__main__":
    main()
