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
        explicit RandomGraph(size_t size, EdgeProposer& edge_proposer, RNG& rng):
            m_size(size),
            m_state(size),
            m_edge_proposer(edge_proposer),
            m_rng(rng) { }

        const MultiGraph& getState() { return m_state; }
        void setState(const MultiGraph& state) { m_state = state; }
        const int getSize() { return m_size; }
        void copyState(const MultiGraph& state);

        virtual void sampleState() = 0;
        virtual double getLogLikelihood(const MultiGraph& graph) const = 0;
        double getLogLikelihood() const { return getLogLikelihood(m_state); };
        double getLogPrior() const { return 0.; }
        double getLogJoint() const { return getLogLikelihood() + getLogPrior(); }

        GraphMove proposeMove() { return m_edge_proposer(); }
        virtual double getLogJointRatio (const GraphMove& move) const = 0;
        void applyMove(const GraphMove& move);
        void enumerateAllGraphs() const;
        void doMetropolisHastingsStep(double beta=1.0) { };


    protected:
        size_t m_size;
        MultiGraph m_state;
        EdgeProposer& m_edge_proposer;
        RNG m_rng;

};

} // namespace FastMIDyNet

#endif
