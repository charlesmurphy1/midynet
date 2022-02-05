#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/edge_proposer/double_edge_swap.h"


namespace FastMIDyNet {


GraphMove DoubleEdgeSwapProposer::proposeRawMove() const {
    auto edge1 = m_edgeSamplableSet.sample_ext_RNG(rng).first;
    auto edge2 = m_edgeSamplableSet.sample_ext_RNG(rng).first;

    if (edge1 == edge2)
        return GraphMove();

    BaseGraph::Edge newEdge1, newEdge2;
    if (m_swapOrientationDistribution(rng)) {
        newEdge1 = {edge1.first, edge2.first};
        newEdge2 = {edge1.second, edge2.second};
    }
    else {
        newEdge1 = {edge1.first, edge2.second};
        newEdge2 = {edge1.second, edge2.first};
    }
    return {{edge1, edge2}, {newEdge1, newEdge2}};
}

void DoubleEdgeSwapProposer::setUpFromGraph(
    const MultiGraph& graph,
    std::unordered_set<BaseGraph::VertexIndex> blackList
) {
    m_graphPtr = &graph;
    m_edgeSamplableSet.clear();
    for (auto vertex: graph){
        if (blackList.count(vertex) > 0)
            continue;
        for (auto neighbor: graph.getNeighboursOfIdx(vertex))
            if (vertex <= neighbor.vertexIndex
                and blackList.count(neighbor.vertexIndex) == 0
            )
                m_edgeSamplableSet.insert({vertex, neighbor.vertexIndex}, neighbor.label);
    }
}

void DoubleEdgeSwapProposer::updateProbabilities(const GraphMove& move) {
    size_t edgeWeight;
    BaseGraph::Edge edge;
    for (auto removedEdge: move.removedEdges) {
        edge = getOrderedEdge(removedEdge);
        edgeWeight = round(m_edgeSamplableSet.get_weight(edge));
        if (edgeWeight == 1)
            m_edgeSamplableSet.erase(edge);
        else
            m_edgeSamplableSet.set_weight(edge, edgeWeight-1);
    }

    for (auto addedEdge: move.addedEdges) {
        edge = getOrderedEdge(addedEdge);
        if (m_edgeSamplableSet.count(edge) == 0)
            m_edgeSamplableSet.insert(edge, 1);
        else {
            edgeWeight = round(m_edgeSamplableSet.get_weight(edge));
            m_edgeSamplableSet.set_weight(edge, edgeWeight+1);
        }
    }
}


} // namespace FastMIDyNet
