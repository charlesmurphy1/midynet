#ifndef FAST_MIDYNET_BLOCK_COUNT_H
#define FAST_MIDYNET_BLOCK_COUNT_H

#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rng.h"
#include "prior.hpp"


namespace FastMIDyNet{

class BlockCountPrior: public BlockLabeledPrior<size_t> {
public:
    using BlockLabeledPrior<size_t>::BlockLabeledPrior;
    virtual const double getLogLikelihoodFromState(const size_t&) const = 0;
    const double getLogLikelihood() const override { return getLogLikelihoodFromState(m_state); }
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const { return 0; }
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const { throw std::logic_error("BlockCount: this method should not be used."); }
    void setStateFromPartition(const BlockSequence& blocks) { setState(*max_element(blocks.begin(), blocks.end()) + 1);}
protected:
    void _applyGraphMove(const GraphMove& move) override { }
    void _applyLabelMove(const BlockMove& move) override { throw std::logic_error("BlockCount: this method should not be used."); }
    void _samplePriors() override { }
    const double _getLogPrior() const override { return 0; }
    const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const override { return 0; }
    const double _getLogPriorRatioFromLabelMove(const BlockMove& move) const override { return 0; }
};

class BlockCountDeltaPrior: public BlockCountPrior{
public:
    BlockCountDeltaPrior() {}
    BlockCountDeltaPrior(size_t blockCount) { setState(blockCount); }
    BlockCountDeltaPrior(const BlockCountDeltaPrior& other) { setState(other.getState()); }
    virtual ~BlockCountDeltaPrior() {};
    const BlockCountDeltaPrior& operator=(const BlockCountDeltaPrior&other){
        setState(other.m_state);
        return *this;
    }

    void sampleState() override { }

    const double getLogLikelihoodFromState(const size_t& blockCount) const override{
        return (blockCount != m_state) ? -INFINITY : 0;
    }
    void checkSelfConsistency() const override { };

    void checkSelfSafety() const override {
        if (m_state == 0)
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

        const double getMean() const { return m_mean; }
        void setMean(double mean){
            m_mean = mean;
            m_poissonDistribution = std::poisson_distribution<size_t>(mean);
        }
        void sampleState() override;
        const double getLogLikelihoodFromState(const size_t& state) const override;

        void checkSelfConsistency() const override;
};

class BlockCountUniformPrior: public BlockCountPrior{
    size_t m_min, m_max;
    std::uniform_int_distribution<size_t> m_uniformDistribution;

    public:
        BlockCountUniformPrior() {}
        BlockCountUniformPrior(size_t min, size_t max) { setMinMax(min, max); }
        BlockCountUniformPrior(const BlockCountUniformPrior& other) { setMinMax(other.m_min, other.m_max); setState(other.m_state); }
        virtual ~BlockCountUniformPrior() {};
        const BlockCountUniformPrior& operator=(const BlockCountUniformPrior& other) {
            setMin(other.m_min);
            setMax(other.m_max);
            setState(other.m_state);
            return *this;
        }

        const double getMin() const { return m_min; }
        const double getMax() const { return m_max; }
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
        const double getLogLikelihoodFromState(const size_t& state) const override{
            if (state > m_max or state < m_min)
                return -INFINITY;
            return -log(m_max - m_min);
        };

        void checkMin() const;
        void checkMax() const;
        void checkSelfConsistency() const override;
};

class NestedBlockCountPrior: public BlockCountPrior {
protected:
    std::vector<size_t> m_nestedState;
public:
    using BlockCountPrior::BlockCountPrior;
    const double getLogLikelihoodFromState(const size_t&) const override {
         throw std::logic_error("NestedBlockCount: this method should not be used.");
    };
    virtual const double getLogLikelihoodFromNestedState(const std::vector<size_t>&) const = 0;
    const double getLogLikelihood() const override { return getLogLikelihoodFromNestedState(m_nestedState); }

    const size_t getDepth() const { return m_nestedState.size(); }
    const std::vector<size_t>& getNestedState() const { return m_nestedState; }
    void setNestedState(const std::vector<size_t>& nestedBlockCounts) {
        m_nestedState = nestedBlockCounts;
        m_state = nestedBlockCounts[0];
    }
    void setStateFromNestedPartition(const std::vector<std::vector<BlockIndex>>& nestedBlocks) {
        std::vector<size_t> nestedState;
        for (auto b : nestedBlocks)
            nestedState.push_back(*max_element(b.begin(), b.end()) + 1);
        setNestedState(nestedState);
    }
    void checkSelfConsistency() const override {
        if (m_state != m_nestedState[0])
            throw ConsistencyError("NestedBlockCountPrior: m_state (" + std::to_string(m_state)
                                 + ") is inconsistent with m_nestedState[0] (" + std::to_string(m_nestedState[0])
                                 + ")");
    };
};

class NestedBlockCountUniformPrior: public NestedBlockCountPrior{
protected:
    size_t m_graphSize;
public:
    NestedBlockCountUniformPrior(size_t graphSize=1): NestedBlockCountPrior(), m_graphSize(graphSize){}

    void sampleState() override {
        std::vector<size_t> nestedState;
        std::uniform_int_distribution<size_t> dist(1, m_graphSize);
        nestedState.push_back(dist(rng));
        while(nestedState.back() != 1){
            std::uniform_int_distribution<size_t> nestedDist(1, nestedState.back() - 1);
            nestedState.push_back(nestedDist(rng));
        }
        setNestedState(nestedState);
    }

    const double getLogLikelihoodFromNestedState(const std::vector<size_t>& nestedState) const override {
        double logLikelihood = -log(m_graphSize - 1);
        for (size_t l=0; l<nestedState.size()-1; ++l)
            logLikelihood -= log(nestedState[l] - 1);
        return logLikelihood;
    }
    void setGraphSize(size_t size) { m_graphSize = size; }

};

}

#endif
