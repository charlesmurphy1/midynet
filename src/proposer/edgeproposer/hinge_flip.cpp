#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"


namespace FastMIDyNet {


GraphMove HingeFlip::proposeMove() {
    auto edge = m_edgeSamplableSet.sample().first;
    auto node = m_nodeSamplableSet.sample().first;

    if (edge.first == node or edge.second == node)
        return GraphMove();

    BaseGraph::Edge newEdge;
    if (m_flipOrientationDistribution(rng)) {
        newEdge = {edge.first, node};
    }
    else {
        newEdge = {edge.second, node};
    }
    return {{edge}, {newEdge}};
}

void HingeFlip::setup(const RandomGraph& randomGraph) {
    const MultiGraph& graph = randomGraph.getState();
    for (auto vertex: graph) {
        m_nodeSamplableSet.insert(vertex, 1);
        for (auto neighbor: graph.getNeighboursOfIdx(vertex)) {
            if (vertex <= neighbor.vertexIndex)
                m_edgeSamplableSet.insert({vertex, neighbor.vertexIndex}, neighbor.label);
        }
    }
}

void HingeFlip::updateProbabilities(const GraphMove& move) {
    size_t edgeWeight;
    BaseGraph::Edge edge;
    for (auto removedEdge: move.removedEdges) {
        edge = getOrderedEdge(removedEdge);
        edgeWeight = round(m_edgeSamplableSet.get_weight(removedEdge));
        if (edgeWeight == 1)
            m_edgeSamplableSet.erase(removedEdge);
        else
            m_edgeSamplableSet.set_weight(removedEdge, edgeWeight-1);
    }

    for (auto addedEdge: move.addedEdges) {
        edge = getOrderedEdge(addedEdge);
        if (m_edgeSamplableSet.count(addedEdge) == 0)
            m_edgeSamplableSet.insert(addedEdge, 1);
        else {
            edgeWeight = round(m_edgeSamplableSet.get_weight(addedEdge));
            m_edgeSamplableSet.set_weight(addedEdge, edgeWeight+1);
        }
    }
}


} // namespace FastMIDyNet
