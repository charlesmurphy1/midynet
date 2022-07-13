import midynet
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from midynet.util.display import draw_graph, all_color_sequence
from midynet.util.convert import get_networkx_graph_from_basegraph
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    hierarchy.dendrogram(linkage_matrix, **kwargs)


def main():
    N = 100
    E = 500
    B = 4
    config = midynet.config.RandomGraphConfig.planted_partition(N, E, B, 0.3)
    graph = midynet.config.RandomGraphFactory.build(config)

    graph.sample()
    g = graph.get_graph()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    pos = nx.spring_layout(get_networkx_graph_from_basegraph(g))
    draw_graph(g, graph.get_labels(), ax=ax[0], pos=pos)
    ax[0].axis("off")

    clustering = hierarchy.linkage(
        g.get_adjacency_matrix(), method="ward", optimal_ordering=True
    )
    threshold = np.max(clustering[:, 2] - np.roll(clustering[:, 2], 1))
    labels = hierarchy.fcluster(clustering, t=threshold, criterion="distance")
    print(labels)
    draw_graph(g, labels, ax=ax[1], pos=pos)
    ax[1].axis("off")
    hierarchy.dendrogram(clustering, color_threshold=threshold)
    plt.show()


if __name__ == "__main__":
    main()
