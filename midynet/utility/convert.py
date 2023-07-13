import numpy as np
from basegraph import core as bs
import networkx as nx
import importlib


def get_edgelist(bs_graph: bs.UndirectedMultigraph) -> list[tuple[int, int]]:
    # el = []
    # for v in bs_graph:
    #     for u in bs_graph.get_out_neighbours(v):
    #         if v > u.vertex_index:
    #             continue
    #         el.append((v, u.vertex_index))
    return list(bs_graph.edges())


def convert_basegraph_to_networkx(
    bs_graph: bs.UndirectedMultigraph,
) -> nx.Graph:
    nx_graph = nx.Graph()
    for v in bs_graph:
        nx_graph.add_node(v)
        for u in bs_graph.get_out_neighbours(v):
            if v > u:
                continue
            nx_graph.add_edge(v, u)
    return nx_graph


def convert_basegraph_to_graphtool(bs_graph: bs.UndirectedMultigraph):
    if importlib.find_spec("graph_tool"):
        import graph_tool.all as gt
    else:
        raise RuntimeError("Could not find `graph_tool`.")

    gt_graph = gt.Graph(directed=False)
    for e in bs_graph.edges():
        for m in range(bs_graph.get_edge_multiplicity(*e)):
            gt_graph.add_edge(*e)
    return gt_graph


def convert_graphtool_to_basegraph(
    gt_graph, weights=None
) -> bs.UndirectedMultigraph:
    bs_graph = bs.UndirectedMultigraph(gt_graph.num_vertices())
    for i, j in gt_graph.edges():
        if not gt_graph.is_directed() and i > j:
            continue
        bs_graph.add_multiedge(
            i, j, weights[i, j] if weights is not None else 1
        )
    return bs_graph


def reduce_partition(partition: list[int]) -> list[int]:
    remap = {}
    id = 0
    reduced_partition = []
    for b in partition:
        if b not in remap:
            remap[b] = id
            id += 1
        reduced_partition.append(remap[b])
    return reduced_partition


def convert_gt_blockstate_to_partition(block_state) -> list[int]:
    partition = block_state.get_blocks().a
    return reduce_partition(partition)


def save_graph(graph: bs.UndirectedMultigraph, file_name: str) -> None:
    edges = []
    for e in graph.edges():
        edges.append([*e, graph.get_edge_multiplicity(*e)])
    #     for n in graph.get_out_neighbours(i):
    #         if n.vertex_index >= i:
    #             edges.append([i, n.vertex_index, n.label])

    edges = np.array(edges)
    np.save(file_name, edges)


def load_graph(file_name: str, size=None) -> bs.UndirectedMultigraph:
    edges = np.load(file_name)
    graph = bs.UndirectedMultigraph(
        np.max(edges) + 1 if size is None else size
    )
    for i, j, m in edges:
        graph.add_multiedge(i, j, m)
    return graph
