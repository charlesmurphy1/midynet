#include "FastMIDyNet/utility.h"
#include "FastMIDyNet/proposer/edge_proposer/double_edge_swap.h"


namespace FastMIDyNet {


GraphMove DoubleEdgeSwap::proposeMove() {
    auto edge1 = m_edgeSamplableSet.sample().first;
    auto edge2 = m_edgeSamplableSet.sample().first;

    if (edge1 == edge2)
        return GraphMove();

    Edge newEdge1, newEdge2;
    if (m_swapOrientationDistribution(rng)) {
        newEdge1 = {{edge1.first, edge2.first}};
        newEdge2 = {{edge1.second, edge2.second}};
    }
    else {
        newEdge1 = {{edge1.first, edge2.second}};
        newEdge2 = {{edge1.second, edge2.first}};
    }
    return {{edge1, edge2}, {newEdge1, newEdge2}};
}

void DoubleEdgeSwap::setup(const MultiGraph& graph) {
    for (auto vertex: graph)
        for (auto neighbor: graph.getNeighboursOfIdx(vertex))
            if (vertex <= neighbor.first)
                m_edgeSamplableSet.insert({{vertex, neighbor.first}}, neighbor.second);
}

void DoubleEdgeSwap::updateProbabilities(const GraphMove& move) {
    size_t edgeWeight;
    for (auto removedEdge: move.removedEdges) {
        edgeWeight = round(m_edgeSamplableSet.get_weight(removedEdge));
        if (edgeWeight == 1)
            m_edgeSamplableSet.erase(removedEdge);
        else
            m_edgeSamplableSet.set_weight(removedEdge, edgeWeight-1);
    }

    for (auto addedEdge: move.addedEdges) {
        if (m_edgeSamplableSet.count(addedEdge) == 0)
            m_edgeSamplableSet.insert(addedEdge, 1);
        else {
            edgeWeight = round(m_edgeSamplableSet.get_weight(addedEdge));
            m_edgeSamplableSet.set_weight(addedEdge, edgeWeight+1);
        }
    }
}


} // namespace FastMIDyNet
