#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/edge_proposer/double_edge_swap.h"


namespace FastMIDyNet {


GraphMove DoubleEdgeSwapProposer::proposeRawMove() const {
    auto edge1 = m_edgeSampler.sample();
    auto edge2 = m_edgeSampler.sample();

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



    if ( m_edgeSampler.contains(edge1) and m_graphPtr->getEdgeMultiplicityIdx(edge1) == 0)
        throw std::logic_error("DoubleEdgeSwapProposer: Edge ("
                                + std::to_string(edge1.first) + ", "
                                + std::to_string(edge1.second)
                                + ") exists in sampler with multiplicity 0 in graph.");

    if ( m_edgeSampler.contains(edge2) and m_graphPtr->getEdgeMultiplicityIdx(edge2) == 0)
        throw std::logic_error("DoubleEdgeSwapProposer: Edge ("
                                + std::to_string(edge2.first) + ", "
                                + std::to_string(edge2.second)
                                + ") exists in sampler with multiplicity 0 in graph.");

    return {{edge1, edge2}, {newEdge1, newEdge2}};
}

void DoubleEdgeSwapProposer::setUpFromGraph( const MultiGraph& graph ) {
    m_graphPtr = &graph;
    for (auto vertex : graph)
        for (auto neighbor : graph.getNeighboursOfIdx(vertex))
            if (vertex <= neighbor.vertexIndex)
                m_edgeSampler.onEdgeInsertion({vertex, neighbor.vertexIndex}, neighbor.label);
}

void DoubleEdgeSwapProposer::applyGraphMove(const GraphMove& move) {
    for (auto edge: move.removedEdges) {
        edge = getOrderedEdge(edge);
        m_edgeSampler.onEdgeRemoval(edge);
    }
    for (auto edge: move.addedEdges) {
        edge = getOrderedEdge(edge);
        m_edgeSampler.onEdgeAddition(edge);
    }
}


} // namespace FastMIDyNet
