#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/edge_proposer/single_edge.h"


namespace FastMIDyNet {


const GraphMove SingleEdgeProposer::proposeRawMove() const {
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
    for (auto vertex : graph){
        m_vertexSamplerPtr->onVertexInsertion(vertex);
        for (auto neighbor : graph.getNeighboursOfIdx(vertex)){
            if (vertex <= neighbor.vertexIndex)
                m_vertexSamplerPtr->onEdgeInsertion({vertex, neighbor.vertexIndex}, neighbor.label);

        }
    }
}

const double SingleEdgeUniformProposer::getLogProposalProbRatio(const GraphMove&move) const {
    double logProbability = 0;

    for (auto edge: move.removedEdges)
        if (m_graphPtr->getEdgeMultiplicityIdx(edge) == 1)
            logProbability += log(.5);

    for (auto edge: move.addedEdges)
        if (m_graphPtr->getEdgeMultiplicityIdx(edge) == 0)
            logProbability -= log(.5);
    return logProbability;
}

void SingleEdgeDegreeProposer::applyGraphMove(const GraphMove& move) {
    for (auto edge: move.removedEdges)
        m_vertexDegreeSampler.onEdgeRemoval(edge);
    for (auto edge: move.addedEdges)
        m_vertexDegreeSampler.onEdgeAddition(edge);
}

const double SingleEdgeDegreeProposer::getGammaRatio(BaseGraph::Edge edge, const double difference) const{
    double gamma = 0;
    gamma += log(m_vertexDegreeSampler.getVertexWeight(edge.first) + difference);
    gamma += log(m_vertexDegreeSampler.getVertexWeight(edge.second) + difference);
    gamma -= log(m_vertexDegreeSampler.getTotalWeight()  + difference);

    gamma -= log(m_vertexDegreeSampler.getVertexWeight(edge.first));
    gamma -= log(m_vertexDegreeSampler.getVertexWeight(edge.second));
    gamma += log(m_vertexDegreeSampler.getTotalWeight());

    return gamma;
}

const double SingleEdgeDegreeProposer::getLogProposalProbRatio(const GraphMove&move) const {
    double logRatio = 0;

    for (auto edge: move.removedEdges){
        logRatio += getGammaRatio(edge, -1);
        if (m_graphPtr->getEdgeMultiplicityIdx(edge) == 1)
            logRatio += -log(.5);
    }
    for (auto edge: move.addedEdges){
        logRatio += getGammaRatio(edge, 1);
        if (m_graphPtr->getEdgeMultiplicityIdx(edge) == 0)
            logRatio += -log(.5);
    }
    return logRatio;
}

} // namespace FastMIDyNet
