#ifndef FAST_MIDYNET_DOUBLE_EDGE_SWAP_H
#define FAST_MIDYNET_DOUBLE_EDGE_SWAP_H


#include "edge_proposer.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"


namespace FastMIDyNet {

class DoubleEdgeSwap: public EdgeProposer {
    sset::SamplableSet<BaseGraph::Edge> m_edgeSamplableSet = sset::SamplableSet<BaseGraph::Edge>(1, 100);
    std::bernoulli_distribution m_swapOrientationDistribution = std::bernoulli_distribution(.5);

    public:
        GraphMove proposeMove();
        void setUp(const RandomGraph& randomGraph) { setUp(randomGraph.getState()); }
        void setUp(const MultiGraph&);
        double getLogProposalProbRatio(const GraphMove&) const { return 0; }
        void updateProbabilities(const GraphMove&);

        const sset::SamplableSet<BaseGraph::Edge>& getSamplableSet() { return m_edgeSamplableSet; }
};


} // namespace FastMIDyNet


#endif
