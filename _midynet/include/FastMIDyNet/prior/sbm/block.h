#ifndef FAST_MIDYNET_BLOCK_H
#define FAST_MIDYNET_BLOCK_H

#include <vector>
#include <iostream>
#include <memory>

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class BlockPrior: public Prior<BlockSequence>{
private:
    void moveVertexCountsInBlocks(const BlockMove& move);
protected:
    size_t m_size;
    BlockCountPrior* m_blockCountPriorPtr = nullptr;
    std::vector<size_t> m_vertexCountsInBlocks;
public:
    /* Constructors */

    BlockPrior(size_t size, BlockCountPrior& blockCountPrior):
        m_size(size) {  setBlockCountPrior(blockCountPrior); }
    BlockPrior(): m_size(0){}
    BlockPrior(const BlockPrior& other) {
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
        m_size = blocks.size();
        m_vertexCountsInBlocks = computeVertexCountsInBlocks(blocks);
        m_blockCountPriorPtr->setStateFromPartition(blocks);
        m_state = blocks;
    }

    /* Accessors & mutators of attributes */
    const size_t& getSize() const { return m_size; }
    void setSize(size_t size) { m_size = size; }

    /* Accessors & mutators of accessory states */
    const BlockCountPrior& getBlockCountPrior() const { return *m_blockCountPriorPtr; }
    BlockCountPrior& getBlockCountPriorRef() const { return *m_blockCountPriorPtr; }
    void setBlockCountPrior(BlockCountPrior& blockCountPrior) {
        m_blockCountPriorPtr = &blockCountPrior;
        m_blockCountPriorPtr->isRoot(false);
    }

    const size_t& getBlockCount() const { return m_blockCountPriorPtr->getState(); }
    const std::vector<size_t>& getVertexCountsInBlocks() const { return m_vertexCountsInBlocks; };
    const BlockIndex& getBlockOfIdx(BaseGraph::VertexIndex idx) const { return m_state[idx]; }
    static std::vector<size_t> computeVertexCountsInBlocks(const BlockSequence&);

    /* sampling methods */
    // void sampleState() override ;
    void samplePriors() override { m_blockCountPriorPtr->sample(); }

    /* MCMC methods */
    const double getLogPrior() const override { return m_blockCountPriorPtr->getLogJoint(); };

    virtual const double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const = 0;
    const double getLogPriorRatioFromBlockMove(const BlockMove& move) const {
        return m_blockCountPriorPtr->getLogJointRatioFromBlockMove(move);
    }




    const double _getLogJointRatioFromGraphMove(const GraphMove& move) const override { return 0; };
    const double _getLogJointRatioFromBlockMove(const BlockMove& move) const override {
        return getLogLikelihoodRatioFromBlockMove(move) + getLogPriorRatioFromBlockMove(move);
    };

    void _applyGraphMove(const GraphMove&) override { };
    void _applyBlockMove(const BlockMove& move) override {
        m_blockCountPriorPtr->applyBlockMove(move);
        applyBlockMoveToVertexCounts(move);
        applyBlockMoveToState(move);
    }
    void applyBlockMoveToState(const BlockMove& move) { m_state[move.vertexIdx] = move.nextBlockIdx; };
    void applyBlockMoveToVertexCounts(const BlockMove& move) {
        if (move.addedBlocks == 1) m_vertexCountsInBlocks.push_back(0);
        else if (move.addedBlocks == -1) m_vertexCountsInBlocks.erase(m_vertexCountsInBlocks.begin() + move.prevBlockIdx);

        --m_vertexCountsInBlocks[move.prevBlockIdx];
        ++m_vertexCountsInBlocks[move.nextBlockIdx];
    };

    /* Consistency methods */
    static void checkBlockSequenceConsistencyWithBlockCount(const BlockSequence& blockSeq, size_t expectedBlockCount) ;
    static void checkBlockSequenceConsistencyWithVertexCountsInBlocks(const BlockSequence& blockSeq, std::vector<size_t> expectedVertexCountsInBlocks) ;


    void computationFinished() const override {
        m_isProcessed=false;
        m_blockCountPriorPtr->computationFinished();
    }

    void checkSelfConsistency() const override {
        checkBlockSequenceConsistencyWithBlockCount(m_state, getBlockCount());
        checkBlockSequenceConsistencyWithVertexCountsInBlocks(m_state, getVertexCountsInBlocks());
    }

    void checkSelfSafety() const override {
        if (m_size < 0)
            throw SafetyError("BlockPrior: unsafe prior since `size` < 0: " + std::to_string(m_size) + ".");
        if (m_blockCountPriorPtr == nullptr)
            throw SafetyError("BlockUniformPrior: unsafe prior since `m_blockCountPriorPtr` is empty.");

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
    const double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
        return (move.prevBlockIdx != move.nextBlockIdx) ? -INFINITY : 0;
    }
};


class BlockUniformPrior: public BlockPrior{
public:
    using BlockPrior::BlockPrior;
    void sampleState() override ;
    const double getLogLikelihood() const override ;
    const double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const ;
};

class BlockUniformHyperPrior: public BlockPrior{
public:
    using BlockPrior::BlockPrior;
    void sampleState() override ;
    const double getLogLikelihood() const override ;
    const double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const ;
};

}

#endif
