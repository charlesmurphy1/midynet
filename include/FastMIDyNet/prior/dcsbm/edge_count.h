#ifndef FAST_MIDYNET_EDGE_COUNT_H
#define FAST_MIDYNET_EDGE_COUNT_H

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class EdgeCountPrior: public Prior<size_t> {
    public:
        void samplePriors() {}
        virtual double getLogLikelihood(const size_t&) const = 0;
        double getLogLikelihood() const { return getLogLikelihood(m_state); }
        double getLogPrior() { return 0; }
        double getLogLikelihoodRatio(const GraphMove& move) const {
             return getLogLikelihood(getStateAfterMove(move)) - getLogLikelihood();
        }
        double getLogJointRatio(const GraphMove& move) {
            return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatio(move); }, 0);
        }
        double getLogJointRatio(const BlockMove& move) { return 0; }

        void applyMove(const GraphMove& move) {
            processRecursiveFunction( [&](){ setState(getStateAfterMove(move)); } );
        }
        void applyMove(const BlockMove& move) { }
        size_t getStateAfterMove(const GraphMove&) const;
        size_t getStateAfterMove(const BlockMove&) const { return getState(); }
};

class EdgeCountDeltaPrior: public EdgeCountPrior{
    size_t m_edgeCount;
public:
    EdgeCountDeltaPrior(const size_t& edgeCount): m_edgeCount(edgeCount){ setState(m_edgeCount); }
    void sampleState() { };
    double getLogLikelihood(const size_t& state) const { if (state == m_state) return 0; else return -INFINITY; }
    double getLogLikelihoodRatio(const GraphMove& move) { if (move.addedEdges.size() == move.removedEdges.size()) return 0; else return -INFINITY;}
    void checkSelfConsistency() const { };
};

class EdgeCountPoissonPrior: public EdgeCountPrior{
    double m_mean;
    std::poisson_distribution<size_t> m_poissonDistribution;

    public:
        EdgeCountPoissonPrior(double mean): m_mean(mean), m_poissonDistribution(mean) { }

        void sampleState();
        double getLogLikelihood(const size_t& state) const;

        void checkSelfConsistency() const;
};

}

#endif
