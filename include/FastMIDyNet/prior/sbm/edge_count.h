#ifndef FAST_MIDYNET_EDGE_COUNT_H
#define FAST_MIDYNET_EDGE_COUNT_H

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class EdgeCountPrior: public Prior<size_t> {
    public:

        using Prior::Prior;
        void samplePriors() {}
        virtual double getLogLikelihoodFromState(const size_t&) const = 0;
        virtual double getLogLikelihood() const { return getLogLikelihoodFromState(m_state); }
        // double getLogLikelihood() const { return getLogLikelihood(m_state); }
        double getLogPrior() { return 0; }
        double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
             return getLogLikelihoodFromState(getStateAfterGraphMove(move)) - getLogLikelihood();
        }
        double getLogJointRatioFromGraphMove(const GraphMove& move) {
            return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromGraphMove(move); }, 0);
        }
        double getLogJointRatioFromBlockMove(const BlockMove& move) { return 0; }

        void applyGraphMove(const GraphMove& move) {
            processRecursiveFunction( [&](){ setState(getStateAfterGraphMove(move)); } );
            #if DEBUG
            checkSelfConsistency();
            #endif

        }
        void applyBlockMove(const BlockMove& move) { }
        size_t getStateAfterGraphMove(const GraphMove&) const;
};

class EdgeCountDeltaPrior: public EdgeCountPrior{
    size_t m_edgeCount;
public:
    using EdgeCountPrior::EdgeCountPrior;
    EdgeCountDeltaPrior(const size_t& edgeCount): m_edgeCount(edgeCount){ setState(m_edgeCount); }
    void sampleState() { };
    double getLogLikelihoodFromState(const size_t& state) const { if (state == m_state) return 0.; else return -INFINITY; };
    double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) { if (move.addedEdges.size() == move.removedEdges.size()) return 0; else return -INFINITY;}
    void checkSelfConsistency() const { };
};

class EdgeCountPoissonPrior: public EdgeCountPrior{
    double m_mean;
    std::poisson_distribution<size_t> m_poissonDistribution;

    public:
        using EdgeCountPrior::EdgeCountPrior;
        EdgeCountPoissonPrior(double mean): m_mean(mean), m_poissonDistribution(mean) { }

        void sampleState();
        double getLogLikelihoodFromState(const size_t& state) const;

        void checkSelfConsistency() const;
};

}

#endif
