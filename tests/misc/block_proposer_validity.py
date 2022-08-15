import midynet
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt
from collections import defaultdict

from midynet.util.convert import (
    convert_basegraph_to_graphtool,
    convert_basegraph_to_networkx,
)
from midynet.util.display import draw_graph, med_colors

from _midynet.proposer import BlockMove
from _midynet.proposer.label import RestrictedUniformBlockProposer, RestrictedMixedBlockProposer
from _midynet.utility import sampleRandomNeighbor

def klDiv(p, q):
    p = np.array(p)
    q = np.array(q)
    return np.sum(p * np.log(p / q))

def getGroundTruth(vertex, s, graph, blocks, label_graph, shift=1, B=None):
    w, d = 0, 0
    B = np.max(blocks)+1 if B is None else B

    r = blocks[vertex]
    for neighbor in graph.get_out_edges_of_idx(vertex):
        t = blocks[neighbor.vertex_index]
        m = neighbor.label
        if (vertex == neighbor.vertex_index):
            m *= 2
        est = label_graph.get_edge_multiplicity_idx(s, t) if s < label_graph.get_size() else 0
        et = label_graph.get_degree_of_idx(t)
        if (s == t):
            est *= 2
        w += m * (est + shift) / (et + shift * B)
        d += m
    return np.log(w) - np.log(d)

def main():
    N = 100
    E = 200
    B = 20
    c = midynet.config.RandomGraphConfig.planted_partition(
        size=N,
        edge_count=E,
        block_count=B,
        assortativity=0.9,
        stub_labeled=True,
    )
    test_graph = midynet.config.RandomGraphFactory.build(c)

    graph_prior = midynet.random_graph.StochasticBlockModelFamily(
        N,
        E,
        B,
        block_hyperprior=False,
        canonical=False,
        stub_labeled=True,
        block_proposer_type="mixed",
    )
    g = test_graph.get_state()
    # print(f"{g.get_total_edge_number()=}")
    # g.remove_selfloops()
    # g.remove_multiedges()
    # print(f"{g.get_total_edge_number()=}")
    graph_prior.set_state(g)
    g = convert_basegraph_to_graphtool(g)
    ers = np.array(graph_prior.get_label_graph().get_adjacency_matrix())

    gt_prior = gt.BlockState(g, b=g.new_vp("int", vals=graph_prior.get_labels()))
    for n in graph_prior.get_state().get_out_edges_of_idx(0):
        r = graph_prior.get_label_of_idx(0)
        t = graph_prior.get_label_of_idx(n.vertex_index)
        ert = graph_prior.get_label_graph().get_edge_multiplicity_idx(r, t)
        et = graph_prior.get_label_graph().get_degree_of_idx(t)
    # graph_prior.set_labels(test_graph.get_labels())
    p = 0.

    shift = [1] #np.logspace(-3, 4, 20)
    klUni = {}
    klMix = {}

    for s in shift:
        proposer = RestrictedMixedBlockProposer(p, shift=s)

        # block_proposer = RestrictedUniformBlockProposer(p)
        proposer.set_up_with_prior(graph_prior)

        num_draw = 100_000
        gt_move_count = defaultdict(int)
        move_count = defaultdict(int)
        moves = set()



        for i in range(num_draw):
            move = proposer.propose_move(0)
            moves.add((move.prev_label, move.next_label))
            move_count[(move.prev_label, move.next_label)]+=1

            m = gt_prior.sample_vertex_move(0, c=s, d=p)
            moves.add((move.prev_label, m))
            gt_move_count[(move.prev_label, m)]+=1
        print(move_count)
        print(gt_move_count)

        exp_prob = []
        act_prob = []
        gt_exp_prob = []
        gt_act_prob = []
        true = []
        ax = plt.gca()
        for move in moves:
            bmove = BlockMove(0, *move, int(move[1] >= graph_prior.get_label_graph().get_size()))
            print(bmove)
            act_prob.append(move_count[move] / num_draw)
            gt_act_prob.append(gt_move_count[move] / num_draw)

            gtp = gt_prior.get_move_prob(0, move[1], c=s, d=p)
            bsp = proposer.get_log_proposal_prob(bmove)
            tp = getGroundTruth(
                0,
                move[1],
                graph_prior.get_state(),
                graph_prior.get_labels(),
                graph_prior.get_label_graph(),
                shift=s,
                B=B
            )
            print(f"{gtp=}, {bsp=}, {tp=}")
            gt_exp_prob.append(np.exp(gtp))
            exp_prob.append(np.exp(bsp))
            true.append( np.exp(tp) )
        pp = ax.plot(act_prob, marker="o", color=med_colors["red"])
        ax.plot(exp_prob, marker="^", color=med_colors["red"])
        ax.plot(gt_act_prob, marker="o", color=med_colors["blue"])
        ax.plot(gt_exp_prob, marker="^", color=med_colors["blue"])
        ax.plot(true, marker="s", color=med_colors["green"])
        ax.set_xticks(range(len(moves)))
        ax.set_xticklabels([str(m) for m in moves], rotation=45)
        plt.show()

    #     klMix.append(klDiv(exp_mixed_prob, act_prob))
    #     klUni.append(klDiv(exp_uniform_prob, act_prob))
    # plt.loglog(shift, klMix, label="mixed")
    # plt.loglog(shift, klUni, label="uniform")
    # plt.show()
    # print(f"{moves=}, {act_prob=}, {exp_prob=}, {klDiv(exp_prob, act_prob)=}")
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(exp_prob, marker="^", color=med_colors["red"])
    # ax.plot(act_prob, marker="s", color=med_colors["blue"])
    # ax.set_xticks(range(len(exp_prob)))
    # ax.set_xticklabels(moves, rotation=45)
    #
    # fig.tight_layout()
    # plt.show()

def main2():
    N = 10
    E = 20
    B = 3
    c = midynet.config.RandomGraphConfig.planted_partition(
        size=N,
        edge_count=E,
        block_count=B,
        assortativity=0.9,
        stub_labeled=True,
    )
    test_graph = midynet.config.RandomGraphFactory.build(c)
    g = test_graph.get_state()
    print(g)

    counts = defaultdict(int)
    for i in range(100_000):
        counts[sampleRandomNeighbor(g, 0)] += 1
    print(counts)


if __name__ == "__main__":
    main()
