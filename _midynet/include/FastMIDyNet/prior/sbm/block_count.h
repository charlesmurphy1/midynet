#ifndef FAST_MIDYNET_BLOCK_COUNT_H
#define FAST_MIDYNET_BLOCK_COUNT_H

#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/prior/prior.hpp"


namespace FastMIDyNet{

class BlockCountPrior: public Prior<size_t> {
    public:
        using Prior<size_t>::Prior;
        void samplePriors() override { }
        virtual double getLogLikelihoodFromState(const size_t&) const = 0;
        double getLogLikelihood() const override { return getLogLikelihoodFromState(m_state); }
        double getLogPrior() const override { return 0; }
        double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const { return 0; }
        double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
            return getLogLikelihoodFromState(getStateAfterBlockMove(move)) - getLogLikelihood();
        }

        double getLogPriorRatioFromGraphMove(const GraphMove& move) const { return 0; }
        double getLogPriorRatioFromBlockMove(const BlockMove& move) const { return 0; }

        double getLogJointRatioFromGraphMove(const GraphMove& move) const { return 0; }
        double getLogJointRatioFromBlockMove(const BlockMove& move) const {
            auto ratio = processRecursiveConstFunction<double>( [&]() { return getLogLikelihoodRatioFromBlockMove(move); }, 0);
            return ratio;
        }
        void applyGraphMove(const GraphMove& move) { }
        void applyBlockMove(const BlockMove& move) {
            processRecursiveFunction( [&](){ setState(getStateAfterBlockMove(move)); } );
        }
        size_t getStateAfterBlockMove(const BlockMove&) const;
        void checkSafety() const { }

};

class BlockCountDeltaPrior: public BlockCountPrior{
    size_t m_blockCount;
public:
    BlockCountDeltaPrior() {}
    BlockCountDeltaPrior(size_t blockCount): m_blockCount(blockCount){ setState(m_blockCount); }
    BlockCountDeltaPrior(const BlockCountDeltaPrior& other):m_blockCount(other.m_blockCount){ setState(m_blockCount); }
    virtual ~BlockCountDeltaPrior() {};
    const BlockCountDeltaPrior& operator=(const BlockCountDeltaPrior&other){
        m_blockCount = other.m_blockCount;
        setState(other.m_state);
        return *this;
    }

    void sampleState() override { }

    double getLogLikelihoodFromState(const size_t& blockCount) const override{
        if (blockCount != m_state) return -INFINITY;
        else return 0;
    }
    double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const { if (move.addedBlocks == 0) return 0; else return -INFINITY; }

    void checkSelfConsistency() const override { };

    void checkSafety() const override {
        if (m_blockCount == 0)
            throw SafetyError("BlockCountDeltaPrior: unsafe prior since `m_blockCount` is zero.");
    }

};

class BlockCountPoissonPrior: public BlockCountPrior{
    double m_mean;
    std::poisson_distribution<size_t> m_poissonDistribution;

    public:
        BlockCountPoissonPrior() {}
        BlockCountPoissonPrior(double mean) { setMean(mean); }
        BlockCountPoissonPrior(const BlockCountPoissonPrior& other) { setMean(other.m_mean); setState(other.m_state); }
        virtual ~BlockCountPoissonPrior() {};
        const BlockCountPoissonPrior& operator=(const BlockCountPoissonPrior& other) {
            setMean(other.m_mean);
            setState(other.m_state);
            return *this;
        }

        double getMean() const { return m_mean; }
        void setMean(double mean){
            m_mean = mean;
            m_poissonDistribution = std::poisson_distribution<size_t>(mean);
        }
        void sampleState() override;
        double getLogLikelihoodFromState(const size_t& state) const override;

        void checkSelfConsistency() const override;
};

class BlockCountUniformPrior: public BlockCountPrior{
    size_t m_min, m_max;
    std::uniform_int_distribution<size_t> m_uniformDistribution;

    public:
        BlockCountUniformPrior() {}
        BlockCountUniformPrior(size_t min) { setMin(min); }
        BlockCountUniformPrior(size_t min, size_t max) { setMinMax(min, max); }
        BlockCountUniformPrior(const BlockCountUniformPrior& other) { setMinMax(other.m_min, other.m_max); setState(other.m_state); }
        virtual ~BlockCountUniformPrior() {};
        const BlockCountUniformPrior& operator=(const BlockCountUniformPrior& other) {
            setMin(other.m_min);
            setMax(other.m_max);
            setState(other.m_state);
            return *this;
        }

        double getMin() const { return m_min; }
        double getMax() const { return m_max; }
        void setMin(size_t min){
            m_min = min;
            checkMin();
            m_uniformDistribution = std::uniform_int_distribution<size_t>(m_min, m_max);
        }
        void setMax(size_t max){
            m_max = max;
            checkMax();
            m_uniformDistribution = std::uniform_int_distribution<size_t>(m_min, m_max);
        }
        void setMinMax(size_t min, size_t max){
            setMin(min);
            setMax(max);
        }
        void sampleState() override { setState(m_uniformDistribution(rng)); }
        double getLogLikelihoodFromState(const size_t& state) const override{
            return -log(m_max - m_min);
        };

        void checkMin() const;
        void checkMax() const;
        void checkSelfConsistency() const override;
};

}

#endif