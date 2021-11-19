#ifndef FAST_MIDYNET_EDGE_COUNT_H
#define FAST_MIDYNET_EDGE_COUNT_H

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class EdgeCountPrior: public Prior<size_t> {
    public:
        void samplePriors() {}
        double getLogPrior() { return 0; }
        double getLogLikelihoodRatio(const GraphMove& move) const {
             return getLogLikelihood(getStateAfterMove(move)) - getLogLikelihood();
        }
        double getLogJointRatio(const GraphMove& move) {
            return processRecursiveFunction<double>( [&]() {
                    return getLogLikelihoodRatio(move); },
                    0
                );
        }
        double getLogJointRatio(const BlockMove& move) { return 0; }

        void applyMove(const GraphMove& move) {
            processRecursiveFunction( [&](){ setState(getStateAfterMove(move)); } );
        }
        void applyMove(const BlockMove& move) { }
        size_t getStateAfterMove(const GraphMove&) const;
        size_t getStateAfterMove(const BlockMove&) const { return getState(); }
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
