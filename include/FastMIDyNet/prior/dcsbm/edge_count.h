#ifndef FAST_MIDYNET_EDGE_COUNT_H
#define FAST_MIDYNET_EDGE_COUNT_H

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class EdgeCountPrior: public Prior<size_t> {
    public:
        double getLogPrior() const { return 0; }
        double getLogLikelihoodRatio(const GraphMove& move) const {
             return getLogLikelihood(getStateAfterMove(move)) - getLogLikelihood();
        }
        double getLogJointRatio(const GraphMove& move) {
            double logJointRatio = 0;
            if (!isProcessed)
                logJointRatio = getLogLikelihoodRatio(move);
            isProcessed = true;
            return logJointRatio;
        }
        double getLogJointRatio(const BlockMove& move) { return 0; }

        void applyMove(const GraphMove& move) {
            if (!isProcessed)
                setState(getStateAfterMove(move));
            isProcessed=true;
        }
        void applyMove(const std::vector<BlockMove>& move) { }
        size_t getStateAfterMove(const GraphMove&) const;
        size_t getStateAfterMove(const std::vector<BlockMove>&) const { return getState(); }
};


class EdgeCountPoissonPrior: public EdgeCountPrior{
    double m_mean;
    std::poisson_distribution<size_t> m_poissonDistribution;

    public:
        EdgeCountPoissonPrior(double mean): m_mean(mean), m_poissonDistribution(mean) { }

        size_t sample();
        double getLogLikelihood(const size_t& state) const;

        void checkSelfConsistency() const;
};

}

#endif
