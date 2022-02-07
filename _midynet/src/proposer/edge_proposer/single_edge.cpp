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

} // namespace FastMIDyNet
