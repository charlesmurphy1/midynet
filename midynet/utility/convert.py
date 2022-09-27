import basegraph.core as bs
import networkx as nx


def get_edgelist(bs_graph: bs.UndirectedMultigraph) -> list[tuple[int, int]]:
    el = []
    for v in bs_graph:
        for u in bs_graph.get_out_edges_of_idx(v):
            if v > u.vertex_index:
                continue
            el.append((v, u.vertex_index))
    return el


def convert_basegraph_to_networkx(bs_graph: bs.UndirectedMultigraph) -> nx.Graph:
    nx_graph = nx.Graph()
    for v in bs_graph:
        nx_graph.add_node(v)
        for u in bs_graph.get_out_edges_of_idx(v):
            if v > u.vertex_index:
                continue
            nx_graph.add_edge(v, u.vertex_index)
    return nx_graph


def convert_basegraph_to_graphtool(bs_graph: bs.UndirectedMultigraph):
    if importlib.find_spec("graph_tool"):
        import graph_tool.all as gt
    else:
        raise RuntimeError("Could not find `graph_tool`.")

    gt_graph = gt.Graph(directed=False)
    for v in bs_graph:
        for u in bs_graph.get_out_edges_of_idx(v):
            if v > u.vertex_index:
                continue
            for e in range(u.label):
                gt_graph.add_edge(v, u.vertex_index)
    return gt_graph


def convert_graphtool_to_basegraph(gt_graph) -> bs.UndirectedMultigraph:
    bs_graph = bs.UndirectedMultigraph(gt_graph.num_vertices())
    for e in gt_graph.edges():
        bs_graph.add_edge_idx(*e)
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
