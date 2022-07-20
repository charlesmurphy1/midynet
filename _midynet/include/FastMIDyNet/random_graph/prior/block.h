#ifndef FAST_MIDYNET_BLOCK_H
#define FAST_MIDYNET_BLOCK_H

#include <vector>
#include <iostream>
#include <memory>

#include "prior.hpp"
#include "block_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/utility/functions.h"

namespace FastMIDyNet{

class BlockPrior: public BlockLabeledPrior<BlockSequence>{
protected:
    size_t m_size;
    BlockCountPrior* m_blockCountPriorPtr = nullptr;
    CounterMap<BlockIndex> m_vertexCounts;

    const double _getLogPrior() const override { return m_blockCountPriorPtr->getLogJoint(); };
    void _samplePriors() override { m_blockCountPriorPtr->sample(); }

    void _applyGraphMove(const GraphMove&) override { };
    virtual void _applyLabelMove(const BlockMove& move) override {
        m_blockCountPriorPtr->setState(m_blockCountPriorPtr->getState() + move.addedLabels);
        m_vertexCounts.decrement(move.prevLabel);
        m_vertexCounts.increment(move.nextLabel);
        m_state[move.vertexIndex] = move.nextLabel;
    }


    const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const override { return 0; };
    virtual const double _getLogPriorRatioFromLabelMove(const BlockMove& move) const override {
        size_t B = m_blockCountPriorPtr->getState();
        return m_blockCountPriorPtr->getLogLikelihoodFromState(B + move.addedLabels) - m_blockCountPriorPtr->getLogLikelihoodFromState(B);
    }

    void remapBlockIndex(const std::map<size_t, size_t> indexMap){
        auto newBlocks = m_state;
        for (size_t v=0; v<m_size; ++v){
            newBlocks[v] = indexMap.at(m_state[v]);
        }
        setState(newBlocks);
    }
    virtual void setBlockCountFromPartition(const BlockSequence& blocks) { m_blockCountPriorPtr->setState(getMaxBlockCountFromPartition(blocks)); }

public:
    /* Constructors */

    BlockPrior() {}
    BlockPrior(size_t size, BlockCountPrior& blockCountPrior):
        m_size(size) {  setBlockCountPrior(blockCountPrior); }
    BlockPrior(const BlockPrior& other) {
        setSize(other.m_size);
        setState(other.m_state);
        this->setBlockCountPrior(*other.m_blockCountPriorPtr);
    }
    virtual ~BlockPrior(){}
    const BlockPrior& operator=(const BlockPrior& other){
        setState(other.m_state);
        this->setBlockCountPrior(*other.m_blockCountPriorPtr);
        return *this;
    }

    virtual void setState(const BlockSequence& blocks) override{
        m_vertexCounts = computeVertexCounts(blocks);
        m_blockCountPriorPtr->setStateFromPartition(blocks);
        m_state = blocks;
        m_size = blocks.size();
    }

    /* Accessors & mutators of attributes */
    const size_t getSize() const { return m_size; }
    virtual void setSize(size_t size) { m_size = size; }

    /* Accessors & mutators of accessory states */
    const BlockCountPrior& getBlockCountPrior() const { return *m_blockCountPriorPtr; }
    BlockCountPrior& getBlockCountPriorRef() const { return *m_blockCountPriorPtr; }
    void setBlockCountPrior(BlockCountPrior& blockCountPrior) {
        m_blockCountPriorPtr = &blockCountPrior;
        m_blockCountPriorPtr->isRoot(false);
    }

    const size_t getBlockCount() const { return m_blockCountPriorPtr->getState(); }
    const size_t getMaxBlockCount() const { return getMaxBlockCountFromPartition(m_state); }
    const size_t getMaxBlockCountFromPartition(const BlockSequence& blocks) const { return *max_element(blocks.begin(), blocks.end()) + 1; }
    const size_t getEffectiveBlockCount() const { return m_vertexCounts.size(); }
    const size_t getEffectiveBlockCountFromPartition(const BlockSequence& blocks) const { return computeVertexCounts(blocks).size(); }
    const CounterMap<size_t>& getVertexCounts() const { return m_vertexCounts; };
    const BlockIndex getBlockOfIdx(BaseGraph::VertexIndex idx) const { return m_state[idx]; }
    static CounterMap<size_t> computeVertexCounts(const BlockSequence&);

    /* sampling methods */

    /* MCMC methods */


    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override { return 0; }
    virtual const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const = 0;
    bool creatingNewBlock(const BlockMove& move) const {
        return m_vertexCounts.get(move.nextLabel) == 0;
    };
    bool destroyingBlock(const BlockMove& move) const {
        return move.prevLabel != move.nextLabel and m_vertexCounts.get(move.prevLabel) == 1 ;
    }
    const int getAddedBlocks(const BlockMove& move) const {
        return (int) creatingNewBlock(move) - (int) destroyingBlock(move);
    }

    /* Consistency methods */
    static void checkBlockSequenceConsistencyWithVertexCounts(std::string prefix, const BlockSequence& blockSeq, CounterMap<size_t> expectedVertexCounts) ;

    void computationFinished() const override {
        m_isProcessed=false;
        m_blockCountPriorPtr->computationFinished();
    }

    void checkSelfConsistency() const override {
        m_blockCountPriorPtr->checkConsistency();
        checkBlockSequenceConsistencyWithVertexCounts("BlockPrior", m_state, m_vertexCounts);

        if (m_vertexCounts.size() > getBlockCount()){
            throw ConsistencyError("BlockPrior: vertex counts (size "
            + std::to_string(m_vertexCounts.size()) +
            ") are inconsistent with block count (" + std::to_string(getBlockCount()) +  ").");
        }
    }

    bool isSafe() const override {
        return (m_size != 0) and (m_blockCountPriorPtr != nullptr) and (m_blockCountPriorPtr->isSafe());
    }
    void checkSelfSafety() const override {
        if (m_size == 0)
            throw SafetyError("BlockPrior: unsafe prior since `m_size` is zero.");
        if (m_blockCountPriorPtr == nullptr)
            throw SafetyError("BlockPrior: unsafe prior since `m_blockCountPriorPtr` is empty.");
        m_blockCountPriorPtr->checkSafety();

    }

};

class BlockDeltaPrior: public BlockPrior{
private:
    BlockSequence m_blocks;
    BlockCountDeltaPrior m_blockCountDeltaPrior;
public:
    using BlockPrior::BlockPrior;
    BlockDeltaPrior(){ setBlockCountPrior(m_blockCountDeltaPrior); }
    BlockDeltaPrior(const BlockSequence& blocks):
        m_blocks(blocks) {
            setBlockCountPrior(m_blockCountDeltaPrior);
            setState(m_blocks);
        }

    void sampleState() override { }
    const double getLogLikelihood() const override { return 0; }
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
        return (move.prevLabel != move.nextLabel) ? -INFINITY : 0;
    }
};


class BlockUniformPrior: public BlockPrior{
public:
    using BlockPrior::BlockPrior;
    void sampleState() override ;
    const double getLogLikelihood() const override ;
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const ;
};

class BlockUniformHyperPrior: public BlockPrior{
protected:
    void setBlockCountFromPartition(const BlockSequence& blocks){ m_blockCountPriorPtr->setState(getEffectiveBlockCountFromPartition(blocks)); }
public:
    using BlockPrior::BlockPrior;
    void sampleState() override ;
    const double getLogLikelihood() const override ;
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const ;
};

}

#endif
