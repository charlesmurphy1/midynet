#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/edge_proposer/single_edge.h"


namespace FastMIDyNet {


GraphMove SingleEdgeProposer::proposeRawMove() const {
    auto vertex1 = m_vertexSamplerPtr->sample();
    auto vertex2 = m_vertexSamplerPtr->sample();

    BaseGraph::Edge proposedEdge = {vertex1, vertex2};

    std::cout << "N = " << m_graphPtr->getSize() << std::endl;
    std::cout << "E = " << m_graphPtr->getTotalEdgeNumber() << std::endl;
    std::cout << "v1 = " << vertex1 << std::endl;
    std::cout << "v2 = " << vertex2 << std::endl;
    std::cout << "E(v1,v2) = " << m_graphPtr->getEdgeMultiplicityIdx(vertex1, vertex2) << std::endl;


    if (not m_graphPtr->isEdgeIdx(vertex1, vertex2))
        return {{}, {proposedEdge}};

    if (m_addOrRemoveDistribution(rng))
        return {{}, {proposedEdge}};
    return {{proposedEdge}, {}};
}

void SingleEdgeProposer::setUpFromGraph(const MultiGraph& graph) {
    m_graphPtr = &graph;
    m_vertexSamplerPtr->setUp(graph);
}

const double SingleEdgeUniformProposer::getLogProposalProbRatio(const GraphMove&move) const {
    double logProbability = 0;

    for (auto edge: move.removedEdges)
        if (m_graphPtr->getEdgeMultiplicityIdx(edge) == 1)
            logProbability += -log(.5);

    for (auto edge: move.addedEdges)
        if (m_graphPtr->getEdgeMultiplicityIdx(edge) == 0)
            logProbability += -log(.5);
    return logProbability;
}

void SingleEdgeDegreeProposer::applyGraphMove(const GraphMove& move) {
    for (auto edge: move.removedEdges)
        m_vertexDegreeSampler.removeEdge(edge);
    for (auto edge: move.addedEdges)
        m_vertexDegreeSampler.addEdge(edge);
};
const double SingleEdgeDegreeProposer::getLogProposalProbRatio(const GraphMove&move) const {
    double logProbability = 0;

    for (auto edge: move.removedEdges){
        if (m_graphPtr->getEdgeMultiplicityIdx(edge) == 1)
            logProbability += -log(.5);

        logProbability += log(m_vertexDegreeSampler.getVertexWeight(edge.first) - 1);
        logProbability += log(m_vertexDegreeSampler.getVertexWeight(edge.second) - 1);
        logProbability -= log(m_vertexDegreeSampler.getTotalWeight() - 1);

        logProbability -= log(m_vertexDegreeSampler.getVertexWeight(edge.first));
        logProbability -= log(m_vertexDegreeSampler.getVertexWeight(edge.second));
        logProbability += log(m_vertexDegreeSampler.getTotalWeight());
    }
    for (auto edge: move.addedEdges){
        if (m_graphPtr->getEdgeMultiplicityIdx(edge) == 0)
            logProbability += -log(.5);

        logProbability += log(m_vertexDegreeSampler.getVertexWeight(edge.first) + 1);
        logProbability += log(m_vertexDegreeSampler.getVertexWeight(edge.second) + 1);
        logProbability -= log(m_vertexDegreeSampler.getTotalWeight() + 1);

        logProbability -= log(m_vertexDegreeSampler.getVertexWeight(edge.first));
        logProbability -= log(m_vertexDegreeSampler.getVertexWeight(edge.second));
        logProbability += log(m_vertexDegreeSampler.getTotalWeight());
    }
    return logProbability;
}

} // namespace FastMIDyNet
