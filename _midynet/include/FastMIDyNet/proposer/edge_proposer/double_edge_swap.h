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
protected:
    EdgeSampler m_edgeSampler;
public:
    using EdgeProposer::EdgeProposer;
    GraphMove proposeRawMove() const override;
    void setUpFromGraph(const MultiGraph&) override;
    const double getLogProposalProbRatio(const GraphMove& move) const override {
        auto ij = getOrderedEdge(move.removedEdges[0]), kl = getOrderedEdge(move.removedEdges[1]);
        auto ik = getOrderedEdge(move.addedEdges[0]), jl = getOrderedEdge(move.addedEdges[1]);
        double logProb;
        if ( (ij == kl) and (m_edgeSampler.getEdgeWeight(ij) < 2) )
            logProb = -INFINITY;
        else if ( (ij == ik and kl == jl) or (ij == jl and kl == ik) )
            logProb = 0.;
        else{
            double weight = getLogProposalWeight(move);
            GraphMove reversedMove = getReverseMove(move);
            double reversedWeight = getLogReverseProposalWeight(move);
            logProb = reversedWeight - weight;
        }
        return logProb;
    }
    const double getLogProposalWeight(const GraphMove& move) const {
        auto ij = getOrderedEdge(move.removedEdges[0]);
        auto kl = getOrderedEdge(move.removedEdges[1]);
        double w_ij = m_edgeSampler.getEdgeWeight(ij);
        double w_kl = m_edgeSampler.getEdgeWeight(kl);
        return log(w_ij) + log(w_kl);
    }
    const double getLogReverseProposalWeight(const GraphMove& move) const {
        auto ij = getOrderedEdge(move.addedEdges[0]);
        auto kl = getOrderedEdge(move.addedEdges[1]);
        double w_ij = m_edgeSampler.getEdgeWeight(ij);
        double w_kl = m_edgeSampler.getEdgeWeight(kl);
        if (ij == kl)
            return log(w_ij + 2) + log(w_kl + 2);
        else
            return log(w_ij + 1) + log(w_kl + 1);

    }

    void applyGraphMove(const GraphMove&) override;
    void clear() override { m_edgeSampler.clear(); }
};


} // namespace FastMIDyNet


#endif
