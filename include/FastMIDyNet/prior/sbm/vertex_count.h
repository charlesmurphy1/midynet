#ifndef FAST_MIDYNET_VERTEX_COUNT_H
#define FAST_MIDYNET_VERTEX_COUNT_H

#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/utility/functions.h"

namespace FastMIDyNet{

class VertexCountPrior: public Prior<std::vector<size_t>>{
protected:
    size_t m_size;
    BlockCountPrior& m_blockCountPrior;
    void createBlock(){ m_state.push_back(0); }
    void destroyBlock(const BlockIndex& idx) { m_state.erase(m_state.begin() + idx); }
public:
    VertexCountPrior(size_t size, BlockCountPrior& blockCountPrior): // constructor
        m_size(size), m_blockCountPrior(blockCountPrior) { m_blockCountPrior.isRoot(false); }


    void setState(const std::vector<size_t>& state) {
        m_state = state;
        m_blockCountPrior.setState(state.size());
    }

    const size_t& getSize() const { return m_size; }
    const size_t& getBlockCount() const { return m_blockCountPrior.getState(); }
    BlockCountPrior& getBlockCountPrior() const { return m_blockCountPrior; }

    void samplePriors(){ m_blockCountPrior.sample(); }
    double getLogPrior() {
        return m_blockCountPrior.getLogJoint();
    }

    double getLogLikelihoodRatioFromGraphMove(const GraphMove& ) { return 0; }
    virtual double getLogLikelihoodRatioFromBlockMove(const BlockMove& ) const = 0;

    double getLogPriorRatioFromGraphMove(const GraphMove& move) { return 0; }
    double getLogPriorRatioFromBlockMove(const BlockMove& move) { return m_blockCountPrior.getLogJointRatioFromBlockMove(move); }

    double getLogJointRatioFromGraphMove(const GraphMove& move) { return 0; }

    double getLogJointRatioFromBlockMove(const BlockMove& move) {
        return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromBlockMove(move) + getLogPriorRatioFromBlockMove(move); }, 0.);
    }
    void applyGraphMove(const GraphMove&) { };
    void applyBlockMove(const BlockMove& move) {
        processRecursiveFunction( [&]() {
            applyBlockMoveToState(move);
            m_blockCountPrior.applyBlockMove(move);
        });
    }
    void applyBlockMoveToState(const BlockMove& move) {
        if (move.addedBlocks == 1){ createBlock(); }
        --m_state[move.prevBlockIdx];
        ++m_state[move.nextBlockIdx];
        if (move.addedBlocks == -1){ destroyBlock(move.prevBlockIdx); }
    }
    virtual void computationFinished() { m_isProcessed = false; m_blockCountPrior.computationFinished(); }

};

class VertexCountUniformPrior: public VertexCountPrior{
public:
    using VertexCountPrior::VertexCountPrior;
    void sampleState();

    double getLogLikelihood() const { return getLogLikelihoodFromState(getSize(), getBlockCount()); }
    void checkSelfConsistency() const;

    double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const;
    static size_t getSizeFromState(const std::vector<size_t> state){
        size_t sum = 0;
        for(auto nr : state) sum += nr;
        return sum;
    }

protected:
    double getLogLikelihoodFromState(size_t size, size_t blockCount) const { return -logBinomialCoefficient(size - 1, blockCount - 1);}

};

} // FastMIDyNet

#endif
