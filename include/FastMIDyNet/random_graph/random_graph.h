#ifndef FAST_MIDYNET_RANDOM_GRAPH_H
#define FAST_MIDYNET_RANDOM_GRAPH_H

// #include <random>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/edge_proposer.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{


class RandomGraph{
    public:
        explicit RandomGraph(size_t size, EdgeProposer& edge_proposer, RNG rng):
            m_size(size),
            m_state(size),
            m_edge_proposer(edge_proposer),
            m_rng(rng) { }

        const MultiGraph& getState() { return m_state; }
        // void setState(const MultiGraph& state) { m_state = state; }
        const int getSize() { return m_size; }
        void copyState(const MultiGraph& state);

        virtual const double sampleState() = 0;
        virtual const double getLogLikelihood() = 0;
        const double getLogPrior() { return 0.; }
        const double getLogJoint() { return getLogLikelihood() + getLogPrior(); }

        const GraphMove& proposeMove() { return m_edge_proposer(); }
        virtual const double getLogJointRatio(const GraphMove& move) = 0;
        void applyMove(const GraphMove& move);
        void enumerateAllGraphs() const;

    protected:
        size_t m_size;
        MultiGraph m_state;
        EdgeProposer& m_edge_proposer;
        RNG m_rng;

};

} // namespace FastMIDyNet

#endif
