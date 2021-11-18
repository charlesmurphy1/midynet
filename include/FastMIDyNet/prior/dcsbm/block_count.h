#ifndef FAST_MIDYNET_BLOCK_COUNT_H
#define FAST_MIDYNET_BLOCK_COUNT_H

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class BlockCountPrior: public Prior<size_t> {
    public:
        double getLogLikelihoodRatio(const GraphMove& move) const { return 0; }
        double getLogLikelihoodRatio(const std::vector<BlockMove>& move) const {
            return getLogLikelihood(getStateAfterMove(move)) - Prior::getLogLikelihood();
        }

        void applyMove(const GraphMove& move) { getState(); }
        void applyMove(const std::vector<BlockMove>& move) { setState(getStateAfterMove(move)); }
        size_t getStateAfterMove(const GraphMove&) const { return m_state; };
        size_t getStateAfterMove(const std::vector<BlockMove>&) const;
};


class BlockCountPoissonPrior: public BlockCountPrior{
    double m_mean;
    std::poisson_distribution<size_t> m_poissonDistribution;

    public:
        BlockCountPoissonPrior(double mean): m_mean(mean), m_poissonDistribution(mean) { }

        size_t sample();
        double getLogLikelihood(const size_t& state) const;
        double getLogPrior() const { return 0; }

        void checkSelfConsistency() const;
};

}

#endif
