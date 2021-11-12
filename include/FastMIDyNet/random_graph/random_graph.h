#ifndef FAST_MIDYNET_RANDOM_GRAPH_H
#define FAST_MIDYNET_RANDOM_GRAPH_H

#include <random>
#include <vector>

#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class RandomGraph{


    public:
        explicit RandomGraph(GraphPrior& prior, EdgeProposer& edge_proposer, PriorProposer& prior_proposer, RNGType& rng):
            m_prior(prior),
            m_edge_proposer(edge_proposer),
            m_prior_proposer(prior_proposer),
            m_rng(rng) { }

        const GraphType& getState() { return m_state; }
        void setState(GraphType& state) { return m_state; }

        virtual const double sampleState() = 0;
        virtual const double getLogLikelihood() = 0;
        virtual const double getLogPrior() = 0;
        const double getLogJoint() { return getLogLikelihood() + getLogPrior(); }

        const GraphMove& proposeGraphMove() { return m_edge_proposer(); }
        const PriorMove& proposePriorMove() { return m_prior_proposer(); }
        virtual const double getLogJointRatio(const GraphMove& move) = 0;
        virtual const double getLogJointRatio(const PriorMove& move) = 0;
        void applyMove(const GraphMove& move);
        void applyMove(const PriorMove& move) { m_prior.applyMove(move); }

    protected:
        GraphType m_state;
        GraphPrior& m_prior;
        EdgeProposer& m_edge_proposer;
        PriorProposer& m_prior_proposer;
        RNGType m_rng;

};

} // namespace FastMIDyNet

#endif
