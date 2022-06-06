#ifndef FAST_MIDYNET_VERTEX_COUNT_H
#define FAST_MIDYNET_VERTEX_COUNT_H

#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/utility/functions.h"

namespace FastMIDyNet{

class VertexCountPrior: public Prior<std::vector<size_t>>{
protected:
    size_t m_size;
    BlockCountPrior* m_blockCountPriorPtr= nullptr;
    void createBlock(){ m_state.push_back(0); }
    void destroyBlock(const BlockIndex& idx) { m_state.erase(m_state.begin() + idx); }
public:
    VertexCountPrior(){}
    VertexCountPrior(size_t size, BlockCountPrior& blockCountPrior):
        m_size(size) { setBlockCountPrior(blockCountPrior); }
    VertexCountPrior(const VertexCountPrior& other):
        m_size(other.m_size) { setBlockCountPrior(*other.m_blockCountPriorPtr); }
    virtual ~VertexCountPrior(){}
    const VertexCountPrior& operator=(const VertexCountPrior& other){
        this->m_size = other.m_size;
        this->setBlockCountPrior(*other.m_blockCountPriorPtr);
        return *this;
    }


    void setState(const std::vector<size_t>& state) {
        m_state = state;
        m_blockCountPriorPtr->setState(state.size());
    }

    const BlockCountPrior& getBlockCountPrior() const { return *m_blockCountPriorPtr; }
    BlockCountPrior& getBlockCountPriorRef() const { return *m_blockCountPriorPtr; }
    void setBlockCountPrior(BlockCountPrior& blockCountPrior) {
        m_blockCountPriorPtr = &blockCountPrior;
        m_blockCountPriorPtr->isRoot(false);
    }

    const size_t& getSize() const { return m_size; }
    const size_t& getBlockCount() const { return m_blockCountPriorPtr->getState(); }

    void samplePriors() override{ m_blockCountPriorPtr->sample(); }
    const double getLogPrior() const override{
        return m_blockCountPriorPtr->getLogJoint();
    }

    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& ) const { return 0; }
    virtual const double getLogLikelihoodRatioFromBlockMove(const BlockMove& ) const = 0;

    const double getLogPriorRatioFromGraphMove(const GraphMove& move) const { return 0; }
    const double getLogPriorRatioFromBlockMove(const BlockMove& move) const { return m_blockCountPriorPtr->getLogJointRatioFromBlockMove(move); }

    const double getLogJointRatioFromGraphMove(const GraphMove& move) const { return 0; }

    const double getLogJointRatioFromBlockMove(const BlockMove& move) const {
        return processRecursiveConstFunction<double>( [&]() { return getLogLikelihoodRatioFromBlockMove(move) + getLogPriorRatioFromBlockMove(move); }, 0.);
    }
    void applyGraphMove(const GraphMove&) { };
    void applyBlockMove(const BlockMove& move) {
        processRecursiveFunction( [&]() {
            applyBlockMoveToState(move);
            m_blockCountPriorPtr->applyBlockMove(move);
        });
    }
    void applyBlockMoveToState(const BlockMove& move) {
        if (move.addedBlocks == 1){ createBlock(); }
        --m_state[move.prevBlockIdx];
        ++m_state[move.nextBlockIdx];
        if (move.addedBlocks == -1){ destroyBlock(move.prevBlockIdx); }
    }
    virtual void computationFinished() const override { m_isProcessed = false; m_blockCountPriorPtr->computationFinished(); }
    virtual void _checkSafety() const override{
        if (m_size < 0)
            throw SafetyError("BlockPrior: unsafe prior since `size` < 0: " + std::to_string(m_size) + ".");

        if (m_blockCountPriorPtr == nullptr)
            throw SafetyError("BlockUniformPrior: unsafe prior since `m_blockCountPriorPtr` is empty.");

    }

};

class VertexCountUniformPrior: public VertexCountPrior{
public:
    using VertexCountPrior::VertexCountPrior;
    void sampleState() override;

    const double getLogLikelihood() const override { return getLogLikelihoodFromState(getSize(), getBlockCount()); }
    void _checkSelfConsistency() const override;

    const double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const override;
    static size_t getSizeFromState(const std::vector<size_t> state){
        size_t sum = 0;
        for(auto nr : state) sum += nr;
        return sum;
    }

protected:
    const double getLogLikelihoodFromState(size_t size, size_t blockCount) const { return -logBinomialCoefficient(size - 1, blockCount - 1);}

};

} // FastMIDyNet

#endif
