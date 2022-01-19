#ifndef FAST_MIDYNET_DOUBLE_EDGE_SWAP_H
#define FAST_MIDYNET_DOUBLE_EDGE_SWAP_H


#include "edge_proposer.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"


namespace FastMIDyNet {

class DoubleEdgeSwapProposer: public EdgeProposer {
private:
    mutable std::bernoulli_distribution m_swapOrientationDistribution = std::bernoulli_distribution(.5);
protected:
    sset::SamplableSet<BaseGraph::Edge> m_edgeSamplableSet = sset::SamplableSet<BaseGraph::Edge>(1, 100);
public:
    using EdgeProposer::EdgeProposer;
    GraphMove proposeRawMove() const override;
    void setUp(const RandomGraph& randomGraph) override { setUp(randomGraph.getGraph()); }
    void setUp(const MultiGraph&);
    double getLogProposalProbRatio(const GraphMove&) const override { return 0; }
    void updateProbabilities(const GraphMove&) override;

    const sset::SamplableSet<BaseGraph::Edge>& getSamplableSet() { return m_edgeSamplableSet; }
};


} // namespace FastMIDyNet


#endif
