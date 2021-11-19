#ifndef FAST_MIDYNET_BLOCK_COUNT_H
#define FAST_MIDYNET_BLOCK_COUNT_H

#include <limits>
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class BlockCountPrior: public Prior<size_t> {
    public:
        void samplePriors() { }

        double getLogLikelihoodRatio(const GraphMove& move) const { return 0; }
        double getLogLikelihoodRatio(const BlockMove& move) const {
            return getLogLikelihood(getStateAfterMove(move)) - getLogLikelihood();
        }

        double getLogJointRatio(const GraphMove& move) { return 0; }
        double getLogJointRatio(const BlockMove& move) {
            return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatio(move); }, 0);
        }

        double getLogPrior() { return 0; }

        void applyMove(const GraphMove& move) { }
        void applyMove(const BlockMove& move) {
            processRecursiveFunction( [&](){ setState(getStateAfterMove(move)); } );
        }

        size_t getStateAfterMove(const GraphMove&) const { return m_state; };
        size_t getStateAfterMove(const BlockMove&) const;
};

class BlockCountDeltaPrior: public BlockCountPrior{
    size_t m_blockCount;
public:
    BlockCountDeltaPrior(size_t blockCount): m_blockCount(blockCount){ setState(m_blockCount); }
    void sampleState() { }
    double getLogLikelihood(const size_t& blockCount) const{
        if (blockCount != m_state) return -INFINITY;
        else return 0;
}
    double getLogLikelihoodRatio(const BlockMove& move) const {
        if (move.nextBlockIdx >= m_state) return -INFINITY;
        else return 0.;
    }
    double getLogPriorRatio(const BlockMove& move) { return 0.;}
    double getLogJointRatio(const BlockMove& move) {
        return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatio(move); }, 0);
    }

    void checkSelfConsistency() const{ };

};

class BlockCountPoissonPrior: public BlockCountPrior{
    double m_mean;
    std::poisson_distribution<size_t> m_poissonDistribution;

    public:
        BlockCountPoissonPrior(double mean): m_mean(mean), m_poissonDistribution(mean) { }

        void sampleState();
        double getLogLikelihood(const size_t& state) const;

        void checkSelfConsistency() const;
};

}

#endif
