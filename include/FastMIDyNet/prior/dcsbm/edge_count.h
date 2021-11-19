#ifndef FAST_MIDYNET_EDGE_COUNT_H
#define FAST_MIDYNET_EDGE_COUNT_H

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class EdgeCountPrior: public Prior<size_t> {
    public:
        double getLogPrior() { return 0; }
        double getLogLikelihoodRatio(const GraphMove& move) const {
             return getLogLikelihood(getStateAfterMove(move)) - getLogLikelihood();
        }
        double getLogJointRatio(const GraphMove& move) {
            double logJointRatio = 0;
            if (!m_isProcessed)
                logJointRatio = getLogLikelihoodRatio(move);
            m_isProcessed = true;
            return logJointRatio;
        }
        double getLogJointRatio(const BlockMove& move) { return 0; }
        double getLogJointRatio(const MultiBlockMove& move) { return 0; }

        void sampleState() {
            setState(sample());
        }
        void applyMove(const GraphMove& move) {
            if (!m_isProcessed)
                setState(getStateAfterMove(move));
            m_isProcessed=true;
        }
        void applyMove(const BlockMove& move) { }
        void applyMove(const MultiBlockMove& move) { }
        size_t getStateAfterMove(const GraphMove&) const;
        size_t getStateAfterMove(const BlockMove&) const { return getState(); }
        size_t getStateAfterMove(const MultiBlockMove&) const { return getState(); }
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
