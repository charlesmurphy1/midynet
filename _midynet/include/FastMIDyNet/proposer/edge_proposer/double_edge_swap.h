#ifndef FAST_MIDYNET_DOUBLE_EDGE_SWAP_H
#define FAST_MIDYNET_DOUBLE_EDGE_SWAP_H

#include "edge_proposer.h"
#include "FastMIDyNet/proposer/sampler/edge_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"


namespace FastMIDyNet {

class DoubleEdgeSwapProposer: public EdgeProposer {
private:
    mutable std::bernoulli_distribution m_swapOrientationDistribution = std::bernoulli_distribution(.5);
    bool isTrivialMove(const GraphMove&) const;
    bool isHingeMove(const GraphMove&) const;
    const double getLogPropForNormalMove(const GraphMove& move) const ;
    const double getLogPropForDoubleLoopyMove(const GraphMove& move) const ;
    const double getLogPropForDoubleEdgeMove(const GraphMove& move) const ;

protected:
    mutable EdgeSampler m_edgeSampler;
public:
    using EdgeProposer::EdgeProposer;
    const GraphMove proposeRawMove() const override;
    void setUpFromGraph(const MultiGraph&) override;
    const double getLogProposalProbRatio(const GraphMove& move) const override ;

    void applyGraphMove(const GraphMove&) override;
    void clear() override { m_edgeSampler.clear(); }
};


} // namespace FastMIDyNet


#endif
