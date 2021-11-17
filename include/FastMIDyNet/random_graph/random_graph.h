#ifndef FAST_MIDYNET_RANDOM_GRAPH_H
#define FAST_MIDYNET_RANDOM_GRAPH_H

// #include <random>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/edge_proposer.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{


class RandomGraph{
    public:
        explicit RandomGraph(size_t size):
            m_size(size),
            m_state(size) { }

        const MultiGraph& getState() const { return m_state; }
        void setState(const MultiGraph& state) { m_state = state; }
        const int getSize() { return m_size; }
        void copyState(const MultiGraph& state);

        virtual void sampleState() = 0;
        double getLogLikelihood() const { return 0.; };
        double getLogPrior() const { return 0.; }
        double getLogJoint() const { return getLogLikelihood() + getLogPrior(); }

        virtual double getLogJointRatio (const GraphMove& move) const = 0;
        void applyMove(const GraphMove& move);
        void enumerateAllGraphs() const;
        void doMetropolisHastingsStep(double beta=1.0) { };
        void checkConsistency() { };

    protected:
        size_t m_size;
        MultiGraph m_state;

};

} // namespace FastMIDyNet

#endif
