#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"


namespace FastMIDyNet {


GraphMove HingeFlipProposer::proposeRawMove() const {
    auto edge = m_edgeSampler.sample();
    if (m_edgeProposalCounter.count(edge) == 0)
        m_edgeProposalCounter.insert({edge, 0});
    ++m_edgeProposalCounter[edge];

    BaseGraph::VertexIndex vertex = m_vertexSamplerPtr->sample();
    if (m_vertexProposalCounter.count(vertex) == 0)
        m_vertexProposalCounter.insert({vertex, 0});
    ++m_vertexProposalCounter[vertex];

    if (vertex == edge.first)
        return {{{edge.first, edge.second}}, {{vertex, vertex}}};
    if (vertex == edge.second)
        return {{{edge.second, edge.first}}, {{vertex, vertex}}};

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
    // if (edge == newEdge)
    //     return {{}, {}};
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

    // move.display();
    // for (auto vertex : (*m_graphPtr))
    //     for (auto neighbor : m_graphPtr->getNeighboursOfIdx(vertex))
    //         if (vertex <= neighbor.vertexIndex)
    //             std::cout << vertex << ", " << neighbor.vertexIndex << ": " << m_edgeSampler.getEdgeWeight({vertex, neighbor.vertexIndex}) << std::endl;
    //
    // std::cout << std::endl;

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
