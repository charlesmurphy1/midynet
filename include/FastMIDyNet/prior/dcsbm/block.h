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
    virtual double getLogLikelihoodRatio(const MultiBlockMove&) const = 0;
    double getLogLikelihoodRatio(const BlockMove& move) const { return getLogLikelihoodRatio(MultiBlockMove(1, move)); };

    double getLogPriorRatio(const GraphMove& move) { return 0; };
    virtual double getLogPriorRatio(const MultiBlockMove&) = 0;
    double getLogPriorRatio(const BlockMove& move) { return getLogPriorRatio(MultiBlockMove(1, move)); };

    double getLogJointRatio(const GraphMove& move) { return 0; };
    virtual double getLogJointRatio(const MultiBlockMove&) = 0;
    double getLogJointRatio(const BlockMove& move) { return getLogPriorRatio(MultiBlockMove(1, move)); };

    void applyMove(const GraphMove&) { };
    virtual void applyMove(const BlockMove&) = 0;
    void applyMove(const MultiBlockMove& move) { for (auto blockMove: move) applyMove(blockMove); };

};


class BlockUniformPrior: public BlockPrior{
public:
    BlockUniformPrior(size_t size, BlockCountPrior& blockCountPrior):
        BlockPrior(size, blockCountPrior) { }
    BlockSequence sample() ;
    double getLogLikelihood(const BlockSequence& blockSeq) const ;

    double getLogPrior() { return m_blockCountPrior.getLogJoint(); };

    double getLogLikelihoodRatio(const MultiBlockMove&) const;

    double getLogPriorRatio(const MultiBlockMove& move) {
        double logPriorRatio = 0;
        if (!m_isProcessed)
            logPriorRatio = m_blockCountPrior.getLogJointRatio(move);
        m_isProcessed = true;
        return logPriorRatio;
    };

    double getLogJointRatio(const MultiBlockMove& move) { return getLogLikelihoodRatio(move) + getLogPriorRatio(move); };

    void applyMove(const BlockMove&) ;
    static void checkBlockSequenceConsistencyWithBlockCount(const BlockSequence& blockSeq, size_t expectedBlockCount) ;
    void checkSelfConsistency() const { checkBlockSequenceConsistencyWithBlockCount(m_state, getBlockCount()); };

};

class BlockHyperPrior: public BlockPrior{

};

}

#endif
