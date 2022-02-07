#ifndef FAST_MIDYNET_DOUBLE_EDGE_SWAP_H
#define FAST_MIDYNET_DOUBLE_EDGE_SWAP_H

#include "edge_proposer.h"
#include "FastMIDyNet/proposer/edge_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"


namespace FastMIDyNet {

class DoubleEdgeSwapProposer: public EdgeProposer {
private:
    mutable std::bernoulli_distribution m_swapOrientationDistribution = std::bernoulli_distribution(.5);
protected:
    EdgeSampler m_edgeSampler;
public:
    using EdgeProposer::EdgeProposer;
    GraphMove proposeRawMove() const override;
    void setUpFromGraph(const MultiGraph&) override;
    const double getLogProposalProbRatio(const GraphMove&) const override { return 0; }
    void applyGraphMove(const GraphMove&) override;
};


} // namespace FastMIDyNet


#endif
