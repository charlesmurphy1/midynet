#ifndef FAST_MIDYNET_RANDOM_GRAPH_H
#define FAST_MIDYNET_RANDOM_GRAPH_H

// #include <random>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class RandomGraph{
    public:
        RandomGraph(size_t size): m_size(size), m_state(size) { }

        const MultiGraph& getState() const { return m_state; }
        virtual void setState(const MultiGraph& state) { m_state = state; }
        const int getSize() const { return m_size; }

        void sample() {
            samplePriors();
            sampleState();
            #if DEBUG
            checkSelfConsistency();
            #endif
        };
        virtual void sampleState() = 0;
        virtual void samplePriors() = 0;
        virtual double getLogLikelihood() const = 0;
        virtual double getLogPrior() = 0;
        double getLogJoint() { return getLogLikelihood() + getLogPrior(); }

        virtual double getLogLikelihoodRatio (const GraphMove& move) = 0;
        virtual double getLogPriorRatio (const GraphMove& move) = 0;
        double getLogJointRatio (const GraphMove& move){
            return getLogPriorRatio(move) + getLogLikelihoodRatio(move);
        }
        virtual void applyMove(const GraphMove& move);
        // void enumerateAllGraphs() const;
        virtual void checkSelfConsistency() { };

    protected:
        size_t m_size;
        MultiGraph m_state;

};

} // namespace FastMIDyNet

#endif
