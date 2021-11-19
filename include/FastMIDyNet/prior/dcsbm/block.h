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
public:
    BlockPrior(size_t size, BlockCountPrior& blockCountPrior):
        m_size(size), m_blockCountPrior(blockCountPrior) { }

    void setState(const BlockSequence& blockSeq){
        m_state = blockSeq;
        m_blockCountPrior.setState(*max_element(blockSeq.begin(), blockSeq.end()) + 1);
    }

    const size_t& getBlockCount() const { return m_blockCountPrior.getState(); }
    std::vector<size_t> getVertexCount(const BlockSequence& blockSeq) const;
    const size_t& getSize() const { return m_size; }

    double getLogLikelihoodRatio(const GraphMove& move) const { return 0; };
    virtual double getLogLikelihoodRatio(const BlockMove& move) const = 0;

    double getLogPriorRatio(const GraphMove& move) { return 0; };
    virtual double getLogPriorRatio(const BlockMove& move) = 0;

    double getLogJointRatio(const GraphMove& move) { return 0; };
    virtual double getLogJointRatio(const BlockMove& move) = 0;

    void applyMove(const GraphMove&) { };
    virtual void applyMove(const BlockMove&) = 0;

};


class BlockUniformPrior: public BlockPrior{
public:
    BlockUniformPrior(size_t size, BlockCountPrior& blockCountPrior):
        BlockPrior(size, blockCountPrior) { }
    BlockSequence sample() ;
    double getLogLikelihood(const BlockSequence& blockSeq) const ;

    double getLogPrior() { return m_blockCountPrior.getLogJoint(); };

    double getLogLikelihoodRatio(const BlockMove&) const;

    double getLogPriorRatio(const BlockMove& move) {
        return processRecursiveFunction<double>( [&]() {
                return m_blockCountPrior.getLogJointRatio(move); },
                0);
    };

    double getLogJointRatio(const BlockMove& move) {
        return getLogLikelihoodRatio(move) + getLogPriorRatio(move);
    };

    void applyMove(const BlockMove& move) { m_state[move.vertexIdx] = move.nextBlockIdx; };
    static void checkBlockSequenceConsistencyWithBlockCount(const BlockSequence& blockSeq, size_t expectedBlockCount) ;
    void checkSelfConsistency() const {
        checkBlockSequenceConsistencyWithBlockCount(m_state, getBlockCount());
    };

};

class BlockHyperPrior: public BlockPrior{

};

}

#endif
