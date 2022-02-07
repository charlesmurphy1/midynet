#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"


namespace FastMIDyNet {


GraphMove HingeFlipProposer::proposeRawMove() const {
    auto edge = m_edgeSampler.sample();
    BaseGraph::VertexIndex node = m_vertexSamplerPtr->sample();

    if (edge.first == node or edge.second == node)
        return GraphMove();

    BaseGraph::Edge newEdge;
    if (m_flipOrientationDistribution(rng)) {
        newEdge = {edge.first, node};
        edge = {edge.first, edge.second};
    }
    else {
        newEdge = {edge.second, node};
        edge = {edge.second, edge.first};
    }
    return {{edge}, {newEdge}};
};

void HingeFlipProposer::setUpFromGraph(const MultiGraph& graph){
    m_graphPtr = &graph;
    m_edgeSampler.setUp(graph);
    m_vertexSamplerPtr->setUp(graph);
}

void HingeFlipProposer::applyGraphMove(const GraphMove& move) {

    for (auto edge: move.removedEdges) {
        m_vertexSamplerPtr->removeEdge(edge);
        m_edgeSampler.removeEdge(edge);
    }

    for (auto edge: move.addedEdges) {
        m_vertexSamplerPtr->addEdge(edge);
        m_edgeSampler.addEdge(edge);
    }
}


} // namespace FastMIDyNet
