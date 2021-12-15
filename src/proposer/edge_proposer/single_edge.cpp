#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/edge_proposer/single_edge.h"


namespace FastMIDyNet {


GraphMove SingleEdgeProposer::proposeMove() {
    auto vertex1 = m_vertexSamplerPtr->sample();
    auto vertex2 = m_vertexSamplerPtr->sample();
    BaseGraph::Edge proposedEdge = {vertex1, vertex2};

    if (m_graphPtr->getEdgeMultiplicityIdx(vertex1, vertex2) == 0)
        return {{}, {proposedEdge}};

    if (m_addOrRemoveDistribution(rng))
        return {{}, {proposedEdge}};
    return {{proposedEdge}, {}};
}

void SingleEdgeProposer::setUp(const MultiGraph& graph) {
    m_graphPtr = &graph;
    m_vertexSamplerPtr->setUp(graph);
}

double SingleEdgeProposer::getLogProposalProbRatio(const GraphMove& move) const {
    double logProbability = 0;

    for (auto edge: move.removedEdges)
        if (m_graphPtr->getEdgeMultiplicityIdx(edge) == 1)
            logProbability += -log(.5);
    return logProbability;
}

} // namespace FastMIDyNet
