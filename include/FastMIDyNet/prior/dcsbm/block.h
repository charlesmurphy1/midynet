#ifndef FAST_MIDYNET_BLOCK_H
#define FAST_MIDYNET_BLOCK_H

#include <vector>

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/dcsbm/block_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class BlockPrior: public Prior<BlockSequence>{
protected:
    size_t m_size;
    BlockCountPrior& m_blockCountPrior;
    std::vector<size_t> m_vertexCountsInBlocks;
public:
    BlockPrior(size_t size, BlockCountPrior& blockCountPrior):
        m_size(size), m_blockCountPrior(blockCountPrior) { }

    void setState(const BlockSequence& blockSeq) override{
        m_blockCountPrior.setState(*max_element(blockSeq.begin(), blockSeq.end()) + 1);
        m_state = blockSeq;
        m_vertexCountsInBlocks = computeVertexCountsInBlock(m_state);
    }
    void samplePriors() override {
        m_blockCountPrior.sample();
    }
    const size_t& getBlockCount() const { return m_blockCountPrior.getState(); }
    std::vector<size_t> computeVertexCountsInBlock(const BlockSequence&) const;
    const std::vector<size_t>& getVertexCountsInBlock() const { return m_vertexCountsInBlocks; };
    const size_t& getSize() const { return m_size; }

    double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const { return 0; };
    virtual double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const = 0;

    double getLogPriorRatioFromGraphMove(const GraphMove& move) { return 0; };
    virtual double getLogPriorRatioFromBlockMove(const BlockMove& move) = 0;

    double getLogJointRatioFromGraphMove(const GraphMove& move) { return 0; };
    virtual double getLogJointRatioFromBlockMove(const BlockMove& move) = 0;

    void applyGraphMove(const GraphMove&) { };
    void applyBlockMoveLocally(const BlockMove& move) { m_state[move.vertexIdx] = move.nextBlockIdx; };
    virtual void applyBlockMove(const BlockMove&) = 0;
    void computationFinished() override { m_isProcessed=false; m_blockCountPrior.computationFinished(); }

private:
    void destroyBlock();
    void createBlock();
    void moveVertexCountsInBlocks(const BlockMove& move);
};

class BlockDeltaPrior: public BlockPrior{
    BlockSequence m_blockSeq;
    BlockCountDeltaPrior m_blockCountDeltaPrior;
public:
    BlockDeltaPrior(const BlockSequence& blockSeq):
        m_blockSeq(blockSeq),
        m_blockCountDeltaPrior(*max_element(blockSeq.begin(), blockSeq.end())),
        BlockPrior(blockSeq.size(), m_blockCountDeltaPrior) {
            setState(blockSeq);
        }

    void sampleState() {  };
    void samplePriors() { };

    double getLogLikelihood() const { return 0.; }
    double getLogPrior() { return 0.; };

    double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const { if (move.prevBlockIdx != move.nextBlockIdx) return -INFINITY; else return 0.;}
    double getLogPriorRatioFromBlockMove(const BlockMove& move) { return 0; }
    double getLogJointRatioFromBlockMove(const BlockMove& move) {
        return processRecursiveFunction<double>( [&](){ return getLogLikelihoodRatioFromBlockMove(move); }, 0);
    }


    void applyBlockMove(const BlockMove& move){
        processRecursiveFunction( [&]() { applyBlockMoveLocally(move); });
    }

    void checkSelfConsistency() const { };


};

class BlockUniformPrior: public BlockPrior{
public:
    BlockUniformPrior(size_t graphSize, BlockCountPrior& blockCountPrior):
        BlockPrior(graphSize, blockCountPrior) { }

    void sampleState();

    double getLogLikelihood() const ;
    double getLogPrior() { return m_blockCountPrior.getLogJoint(); };

    double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const;
    double getLogPriorRatioFromBlockMove(const BlockMove& move) {
        return m_blockCountPrior.getLogJointRatioFromBlockMove(move);
    };
    double getLogJointRatioFromBlockMove(const BlockMove& move) {
        return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromBlockMove(move) + getLogPriorRatioFromBlockMove(move); }, 0);
    };

    void applyBlockMove(const BlockMove& move){
        processRecursiveFunction( [&]() { m_blockCountPrior.applyBlockMove(move); applyBlockMoveLocally(move); });
    }
    static void checkBlockSequenceConsistencyWithBlockCount(const BlockSequence& blockSeq, size_t expectedBlockCount) ;
    void checkSelfConsistency() const {
        checkBlockSequenceConsistencyWithBlockCount(m_state, getBlockCount());
    };

};

class BlockHyperPrior: public BlockPrior{

};

}

#endif
