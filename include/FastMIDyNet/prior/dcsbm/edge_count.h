#ifndef FAST_MIDYNET_EDGE_COUNT_H
#define FAST_MIDYNET_EDGE_COUNT_H

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class EdgeCountPoissonPrior: public Prior<size_t>{
    double m_mean;
    std::poisson_distribution<size_t> m_poissonDistribution;

    public:
        EdgeCountPoissonPrior(double mean): m_mean(mean), m_poissonDistribution(mean) { }

        size_t sample();
        double getLogLikelihood(size_t state) const;
        double getLogPrior() const { return 0; }
        double getLogLikelihoodRatio(const GraphMove& move) const {
            size_t newState = getStateAfterMove(move);
            return getLogLikelihood(newState) - Prior::getLogLikelihood();
        }

        void applyMove(const GraphMove& move) { setState(getStateAfterMove(move)); }

        void checkSelfConsistency();

    private:
        size_t getStateAfterMove(const GraphMove&) const;
};

}

#endif
