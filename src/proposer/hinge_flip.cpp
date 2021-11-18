#include "FastMIDyNet/utility.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"


namespace FastMIDyNet {


GraphMove HingeFlip::proposeMove() {
    auto edge = m_edgeSamplableSet.sample().first;
    auto node = m_nodeSamplableSet.sample().first;

    if (edge.first == node) or (edge.second == node)
        return GraphMove();

    BaseGraph::Edge newEdge;
    BaseGraph::VertexIndex newNode;
    if (m_flipOrientationDistribution(rng)) {
        newEdge = {edge.first, node};
        newNode = edge.second;
    }
    else {
        newEdge = {edge.second, node};
        newNode = edge.first;
    }
    return {{edge, node}, {newEdge, newNode}};
}

void DoubleEdgeSwap::setup(const MultiGraph& graph) {
    for (auto vertex: graph)
        m_nodeSamplableSet.insert{vertex, 1}
        for (auto neighbor: graph.getNeighboursOfIdx(vertex))
            if (vertex <= neighbor.first)
                m_edgeSamplableSet.insert({vertex, neighbor.first}, neighbor.second);
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
