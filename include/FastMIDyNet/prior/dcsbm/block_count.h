#ifndef FAST_MIDYNET_BLOCK_COUNT_H
#define FAST_MIDYNET_BLOCK_COUNT_H

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class BlockCountPrior: public Prior<size_t> {
    public:
        double getLogLikelihoodRatio(const GraphMove& move) const { return 0; }
        double getLogLikelihoodRatio(const MultiBlockMove& move) const {
            return getLogLikelihood(getStateAfterMove(move)) - Prior::getLogLikelihood();
        }

        double getLogJointRatio(const GraphMove& move) { return 0; }
        double getLogJointRatio(const MultiBlockMove& move) {
            double logJointRatio = 0;
            if (!m_isProcessed)
                logJointRatio = getLogLikelihoodRatio(move);
            m_isProcessed = true;
            return logJointRatio;
        }

        double getLogPrior() { return 0; }

        void applyMove(const GraphMove& move) { }
        void applyMove(const MultiBlockMove& move) {
            if (!m_isProcessed)
                setState(getStateAfterMove(move));
            m_isProcessed = true;
        }

        size_t getStateAfterMove(const GraphMove&) const { return m_state; };
        size_t getStateAfterMove(const BlockMove&) const;
        size_t getStateAfterMove(const MultiBlockMove& move) const {
            size_t newState = getState() ;
            for (auto blockMove: move){
                newState = getStateAfterMove(blockMove) ;
            }
            return newState;
        };
};


class BlockCountPoissonPrior: public BlockCountPrior{
    double m_mean;
    std::poisson_distribution<size_t> m_poissonDistribution;

    public:
        BlockCountPoissonPrior(double mean): m_mean(mean), m_poissonDistribution(mean) { }

        size_t sample();
        double getLogLikelihood(const size_t& state) const;

        void checkSelfConsistency() const;
};

}

#endif
