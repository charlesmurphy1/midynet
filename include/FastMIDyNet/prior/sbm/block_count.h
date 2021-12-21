#ifndef FAST_MIDYNET_BLOCK_COUNT_H
#define FAST_MIDYNET_BLOCK_COUNT_H

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class BlockCountPrior: public Prior<size_t> {
    public:
        void samplePriors() { }
        virtual double getLogLikelihoodFromState(const size_t&) const = 0;
        double getLogLikelihood() const { return getLogLikelihoodFromState(m_state); }
        double getLogPrior() { return 0; }
        double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const { return 0; }
        double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
            return getLogLikelihoodFromState(getStateAfterBlockMove(move)) - getLogLikelihood();
        }
        double getLogJointRatioFromGraphMove(const GraphMove& move) { return 0; }
        double getLogJointRatioFromBlockMove(const BlockMove& move) {
            auto ratio = processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromBlockMove(move); }, 0);
            return ratio;
        }
        void applyGraphMove(const GraphMove& move) { }
        void applyBlockMove(const BlockMove& move) {
            processRecursiveFunction( [&](){ setState(getStateAfterBlockMove(move)); } );
        }
        size_t getStateAfterBlockMove(const BlockMove&) const;

};

class BlockCountDeltaPrior: public BlockCountPrior{
    size_t m_blockCount;
public:
    BlockCountDeltaPrior(size_t blockCount): m_blockCount(blockCount){ setState(m_blockCount); }

    void sampleState() { }

    double getLogLikelihoodFromState(const size_t& blockCount) const{
        if (blockCount != m_state) return -INFINITY;
        else return 0;
    }
    double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) { if (move.addedBlocks == 0) return 0; else return -INFINITY; }

    void checkSelfConsistency() const{ };

};

class BlockCountPoissonPrior: public BlockCountPrior{
    double m_mean;
    std::poisson_distribution<size_t> m_poissonDistribution;

    public:
        BlockCountPoissonPrior(double mean): m_mean(mean), m_poissonDistribution(mean) { }

        void sampleState();
        double getLogLikelihoodFromState(const size_t& state) const;

        void checkSelfConsistency() const;
};

}

#endif
