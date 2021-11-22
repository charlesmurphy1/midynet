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
    std::vector<size_t> m_vertexCountsInBlock;
public:
    BlockPrior(size_t size, BlockCountPrior& blockCountPrior):
        m_size(size), m_blockCountPrior(blockCountPrior) { }

    void setState(const BlockSequence& blockSeq) override{
        m_blockCountPrior.setState(*max_element(blockSeq.begin(), blockSeq.end()) + 1);
        m_state = blockSeq;
        m_vertexCountsInBlock = computeVertexCountsInBlock(m_state);
    }
    void samplePriors() override { m_blockCountPrior.sample(); }
    const size_t& getBlockCount() const { return m_blockCountPrior.getState(); }
    std::vector<size_t> computeVertexCountsInBlock(const BlockSequence&) const;
    const std::vector<size_t>& getVertexCountsInBlock() const { return m_vertexCountsInBlock; };
    const size_t& getSize() const { return m_size; }

    virtual double getLogLikelihood(const BlockSequence&) const = 0;
    double getLogLikelihood() const { return getLogLikelihood(m_state); }

    double getLogLikelihoodRatio(const GraphMove& move) const { return 0; };
    virtual double getLogLikelihoodRatio(const BlockMove& move) const = 0;

    double getLogPriorRatio(const GraphMove& move) { return 0; };
    virtual double getLogPriorRatio(const BlockMove& move) = 0;

    double getLogJointRatio(const GraphMove& move) { return 0; };
    virtual double getLogJointRatio(const BlockMove& move) = 0;

    void applyMove(const GraphMove&) { };
    void applyMoveLocally(const BlockMove& move) { m_state[move.vertexIdx] = move.nextBlockIdx; };
    virtual void applyMove(const BlockMove&) = 0;
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
        BlockPrior(blockSeq.size(), m_blockCountDeltaPrior) { }

    void sampleState() {  };
    void samplePriors() { };

    double getLogLikelihood(const BlockSequence& state) const {
        if (state.size() != getSize()) return -INFINITY;
        for (size_t i = 0; i < state.size(); i++) {
            if (state[i] != m_state[i]) return -INFINITY;
        }
        return 0.;
    };
    double getLogPrior() { return 0.; };

    double getLogLikelihoodRatio(const GraphMove& move) const { return -INFINITY; }
    double getLogLikelihoodRatio(const BlockMove& move) const { return -INFINITY; }
    double getLogPriorRatio(const GraphMove& move) { return -INFINITY; }
    double getLogPriorRatio(const BlockMove& move) { return -INFINITY; }
    double getLogJointRatio(const GraphMove& move) { return -INFINITY; }
    double getLogJointRatio(const BlockMove& move) { return -INFINITY; }


    void applyMove(const BlockMove& move){
        processRecursiveFunction( [&]() { applyMoveLocally(move); });
    }

    void checkSelfConsistency() const { };


};

class BlockUniformPrior: public BlockPrior{
public:
    BlockUniformPrior(size_t graphSize, BlockCountPrior& blockCountPrior):
        BlockPrior(graphSize, blockCountPrior) { }

    void sampleState();

    double getLogLikelihood(const BlockSequence& blockSeq) const ;
    double getLogPrior() { return m_blockCountPrior.getLogJoint(); };

    double getLogLikelihoodRatio(const BlockMove&) const;
    double getLogPriorRatio(const BlockMove& move) {
        return m_blockCountPrior.getLogJointRatio(move);
    };
    double getLogJointRatio(const BlockMove& move) {
        return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatio(move) + getLogPriorRatio(move); }, 0);
    };

    void applyMove(const BlockMove& move){
        processRecursiveFunction( [&]() { m_blockCountPrior.applyMove(move); applyMoveLocally(move); });
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
