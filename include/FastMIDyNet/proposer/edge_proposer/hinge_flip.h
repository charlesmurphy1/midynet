#ifndef FAST_MIDYNET_DOUBLE_EDGE_SWAP_H
#define FAST_MIDYNET_DOUBLE_EDGE_SWAP_H


#include "edge_proposer.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"

namespace FastMIDyNet {

class HingeFlip: public EdgeProposer {
    sset::SamplableSet<BaseGraph::Edge> m_edgeSamplableSet = sset::SamplableSet<BaseGraph::Edge> (1, 2000);
    sset::SamplableSet<BaseGraph::VertexIndex> m_nodeSamplableSet = sset::SamplableSet<BaseGraph::VertexIndex> (1, 2000);
    std::bernoulli_distribution m_flipOrientationDistribution = std::bernoulli_distribution(.5);

    public:
        GraphMove proposeMove();
        void setup(const MultiGraph&);
        double getLogProposalProbRatio(const GraphMove&) const { return 0; }
        void updateProbabilities(const GraphMove&);

        // For tests
        const sset::SamplableSet<BaseGraph::Edge>& getEdgeSamplableSet() { return m_edgeSamplableSet; }
        const sset::SamplableSet<BaseGraph::VertexIndex>& getNodeSamplableSet() { return m_nodeSamplableSet; }
};


} // namespace FastMIDyNet


#endif
