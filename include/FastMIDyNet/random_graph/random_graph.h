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
<<<<<<< HEAD
        explicit RandomGraph(size_t size, RNG& rng):
            m_size(size),
            m_state(size),
            m_rng(rng) { }
=======
        explicit RandomGraph(size_t size, EdgeProposer& edgeProposer):
            m_size(size),
            m_state(size),
            m_edgeProposer(edgeProposer) { }
>>>>>>> main

        const MultiGraph& getState() { return m_state; }
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
<<<<<<< HEAD
        RNG m_rng;
=======
        EdgeProposer& m_edgeProposer;
>>>>>>> main

};

} // namespace FastMIDyNet

#endif
