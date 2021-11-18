#ifndef FAST_MIDYNET_EDGE_COUNT_H
#define FAST_MIDYNET_EDGE_COUNT_H

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class EdgeCountPrior: public Prior<size_t> {
    public:
        double getLogLikelihoodRatio(const GraphMove& move) const {
            return getLogLikelihood(getStateAfterMove(move)) - Prior::getLogLikelihood();
        }
        double getLogLikelihoodRatio(const std::vector<BlockMove>& move) const { return 0; }

        void applyMove(const GraphMove& move) { setState(getStateAfterMove(move)); }
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
        double getLogPrior() const { return 0; }

        void checkSelfConsistency() const;
};

}

#endif
