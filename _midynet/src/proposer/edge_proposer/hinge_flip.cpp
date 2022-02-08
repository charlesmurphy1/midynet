#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"


namespace FastMIDyNet {


GraphMove HingeFlipProposer::proposeRawMove() const {
    auto edge = m_edgeSampler.sample();
    BaseGraph::VertexIndex vertex = m_vertexSamplerPtr->sample();

    if (edge.first == vertex or edge.second == vertex)
        return GraphMove();

    BaseGraph::Edge newEdge;
    if (m_flipOrientationDistribution(rng)) {
        newEdge = {edge.first, vertex};
        edge = {edge.first, edge.second};
    }
    else {
        newEdge = {edge.second, vertex};
        edge = {edge.second, edge.first};
    }
    return {{edge}, {newEdge}};
};

void HingeFlipProposer::setUpFromGraph(const MultiGraph& graph){
    m_graphPtr = &graph;
    for (auto vertex : graph){
        m_vertexSamplerPtr->insertVertex(vertex);
        for (auto neighbor : graph.getNeighboursOfIdx(vertex)){
            if (vertex <= neighbor.vertexIndex){
                m_vertexSamplerPtr->insertEdge({vertex, neighbor.vertexIndex}, neighbor.label);
                m_edgeSampler.insertEdge({vertex, neighbor.vertexIndex}, neighbor.label);
            }
        }
    }
}

void HingeFlipProposer::applyGraphMove(const GraphMove& move) {
    for (auto edge : move.removedEdges){
        m_vertexSamplerPtr->removeEdge(edge);
        m_edgeSampler.removeEdge(edge);
    }
    for (auto edge : move.addedEdges){
        m_vertexSamplerPtr->addEdge(edge);
        m_edgeSampler.addEdge(edge);
    }

}


} // namespace FastMIDyNet
