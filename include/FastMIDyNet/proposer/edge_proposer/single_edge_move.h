#ifndef FAST_MIDYNET_SINGLE_EDGE_MOVE_H
#define FAST_MIDYNET_SINGLE_EDGE_MOVE_H


#include "edge_proposer.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"


namespace FastMIDyNet {

class SingleEdgeMove: public EdgeProposer {
    sset::SamplableSet<BaseGraph::VertexIndex> m_vertexDistribution = sset::SamplableSet<BaseGraph::VertexIndex>(1, 100);
    std::bernoulli_distribution m_addOrRemoveDistribution = std::bernoulli_distribution(.5);
    const FastMIDyNet::MultiGraph* m_graphPtr = NULL;

    public:
        GraphMove proposeMove();
        void setup(const RandomGraph&);
        double getLogProposalProbRatio(const GraphMove&) const;
        void updateProbabilities(const GraphMove&) { }

        const sset::SamplableSet<BaseGraph::VertexIndex>& getSamplableSet() { return m_vertexDistribution; }
};


} // namespace FastMIDyNet


#endif
