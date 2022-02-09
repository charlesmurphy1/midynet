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

    if ( m_edgeSampler.contains(edge) and m_graphPtr->getEdgeMultiplicityIdx(edge) == 0)
        throw std::logic_error("HingeFlipProposer: Edge ("
                                + std::to_string(edge.first) + ", "
                                + std::to_string(edge.second)
                                + ") exists in sampler with multiplicity 0 in graph.");

    return {{edge}, {newEdge}};
};

void HingeFlipProposer::setUpFromGraph(const MultiGraph& graph){
    m_graphPtr = &graph;
    for (auto vertex : graph)
        m_vertexSamplerPtr->onVertexInsertion(vertex);
    for (auto vertex : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(vertex)){
            if (vertex <= neighbor.vertexIndex){
                m_vertexSamplerPtr->onEdgeInsertion({vertex, neighbor.vertexIndex}, neighbor.label);
                m_edgeSampler.onEdgeInsertion({vertex, neighbor.vertexIndex}, neighbor.label);
            }
        }
    }
}

void HingeFlipProposer::applyGraphMove(const GraphMove& move) {
    for (auto edge : move.removedEdges){
        edge = getOrderedEdge(edge);
        m_vertexSamplerPtr->onEdgeRemoval(edge);
        m_edgeSampler.onEdgeRemoval(edge);
    }
    for (auto edge : move.addedEdges){
        edge = getOrderedEdge(edge);
        m_vertexSamplerPtr->onEdgeAddition(edge);
        m_edgeSampler.onEdgeAddition(edge);
    }

}


} // namespace FastMIDyNet
