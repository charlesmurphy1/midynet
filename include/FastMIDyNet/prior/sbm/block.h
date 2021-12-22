#ifndef FAST_MIDYNET_BLOCK_H
#define FAST_MIDYNET_BLOCK_H

#include <vector>
#include <iostream>

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/vertex_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class BlockPrior: public Prior<BlockSequence>{
private:
    void moveVertexCountsInBlocks(const BlockMove& move);
protected:
    size_t m_size;
    size_t m_blockCount;
    std::vector<size_t> m_vertexCountsInBlocks;
public:
    /* Constructors */
    BlockPrior(size_t size): m_size(size){ }
    BlockPrior(): m_size(0){}
    BlockPrior(const BlockPrior& other) { setState(other.m_state);}
    virtual ~BlockPrior(){}
    const BlockPrior& operator=(const BlockPrior& other){ setState(other.m_state); return *this;}

    virtual void setState(const BlockSequence& blocks) override{
        m_size = blocks.size();
        m_blockCount = computeBlockCount(blocks);
        m_vertexCountsInBlocks = computeVertexCountsInBlocks(blocks);
        m_state = blocks;
    }

    /* Accessors & mutators of attributes */
    const size_t& getSize() const { return m_size; }
    void setSize(size_t size) { m_size = size; }

    /* Accessors & mutators of accessory states */
    virtual const size_t& getBlockCount() const { return m_blockCount; }
    virtual const std::vector<size_t>& getVertexCountsInBlocks() const { return m_vertexCountsInBlocks; };
    const BlockIndex& getBlockOfIdx(BaseGraph::VertexIndex idx) const { return m_state[idx]; }
    static size_t computeBlockCount(const BlockSequence& blocks) { return *max_element(blocks.begin(), blocks.end()) + 1; }
    static std::vector<size_t> computeVertexCountsInBlocks(const BlockSequence&);

    /* MCMC methods */
    double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const { return 0; };
    virtual double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const = 0;

    double getLogPriorRatioFromGraphMove(const GraphMove& move) { return 0; };
    virtual double getLogPriorRatioFromBlockMove(const BlockMove& move) = 0;

    double getLogJointRatioFromGraphMove(const GraphMove& move) { return 0; };
    double getLogJointRatioFromBlockMove(const BlockMove& move) {
        return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromBlockMove(move) + getLogPriorRatioFromBlockMove(move); }, 0);
    };

    void applyGraphMove(const GraphMove&) { };
    void applyBlockMoveToState(const BlockMove& move) { m_state[move.vertexIdx] = move.nextBlockIdx; };
    void applyBlockMoveToVertexCounts(const BlockMove& move) {
        if (move.addedBlocks == 1) m_vertexCountsInBlocks.push_back(0);
        else if (move.addedBlocks == -1) m_vertexCountsInBlocks.erase(m_vertexCountsInBlocks.begin() + move.prevBlockIdx);

        --m_vertexCountsInBlocks[move.prevBlockIdx];
        ++m_vertexCountsInBlocks[move.nextBlockIdx];
    };
    virtual void applyBlockMove(const BlockMove&) = 0;
    virtual void computationFinished() override { m_isProcessed=false; }

    /* Consistency methods */
    static void checkBlockSequenceConsistencyWithBlockCount(const BlockSequence& blockSeq, size_t expectedBlockCount) ;
    static void checkBlockSequenceConsistencyWithVertexCountsInBlocks(const BlockSequence& blockSeq, std::vector<size_t> expectedVertexCountsInBlocks) ;

};

class BlockDeltaPrior: public BlockPrior{
    BlockSequence m_blockSeq;
public:
    BlockDeltaPrior(const BlockSequence& blockSeq):
        BlockPrior(blockSeq.size()) { setState(blockSeq); }

    BlockDeltaPrior(){}
    BlockDeltaPrior(const BlockDeltaPrior& blockDeltaPrior):
        BlockPrior(blockDeltaPrior.getSize()) { setState(blockDeltaPrior.getState()); }
    virtual ~BlockDeltaPrior(){}
    const BlockDeltaPrior& operator=(const BlockDeltaPrior& other){
        this->setBlocks(other.getState());
        return *this;
    }


    void setBlocks(const BlockSequence& blocks){
        m_blockSeq = blocks;
        setState(blocks);
    }
    void sampleState() {  };
    void samplePriors() { };

    double getLogLikelihood() const { return 0.; }
    double getLogPrior() { return 0.; };

    double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
        if (move.prevBlockIdx != move.nextBlockIdx) return -INFINITY;
        else return 0.;
    }
    double getLogPriorRatioFromBlockMove(const BlockMove& move) { return 0; }
    double getLogJointRatioFromBlockMove(const BlockMove& move) {
        return processRecursiveFunction<double>( [&](){ return getLogLikelihoodRatioFromBlockMove(move); }, 0);
    }


    void applyBlockMove(const BlockMove& move){
        processRecursiveFunction( [&]() { applyBlockMoveToState(move); });
    }

    void checkSelfConsistency() const { };


};

class BlockUniformPrior: public BlockPrior{
private:
    BlockCountPrior* m_blockCountPriorPtr = NULL;
public:

    BlockUniformPrior(){}
    BlockUniformPrior(size_t graphSize, BlockCountPrior& blockCountPrior):
        BlockPrior(graphSize), m_blockCountPriorPtr(&blockCountPrior) { m_blockCountPriorPtr->isRoot(false); }
    BlockUniformPrior(const BlockUniformPrior& other){
        this->setState(other.m_state);
        this->setBlockCountPrior(*other.m_blockCountPriorPtr);
    }
    virtual ~BlockUniformPrior(){}
    const BlockUniformPrior& operator=(const BlockUniformPrior& other){
        this->setState(other.m_state);
        this->setBlockCountPrior(*other.m_blockCountPriorPtr);
        return *this;
    }

    const BlockCountPrior& getBlockCountPrior() const { return *m_blockCountPriorPtr; }
    BlockCountPrior& getBlockCountPriorRef() const { return *m_blockCountPriorPtr; }
    void setBlockCountPrior(BlockCountPrior& blockCountPrior) {
        m_blockCountPriorPtr = &blockCountPrior; m_blockCountPriorPtr->isRoot(false);
    }
    const size_t& getBlockCount() const { return m_blockCountPriorPtr->getState();}
    void setState(const BlockSequence& blockSeq) override{
        BlockPrior::setState(blockSeq);
        m_blockCountPriorPtr->setState(m_blockCount);
    }
    void sampleState();
    void samplePriors() { m_blockCountPriorPtr->sample(); }

    double getLogLikelihood() const ;
    double getLogPrior() { return m_blockCountPriorPtr->getLogJoint(); };

    double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const;
    double getLogPriorRatioFromBlockMove(const BlockMove& move) {
        return m_blockCountPriorPtr->getLogJointRatioFromBlockMove(move);
    };

    void applyBlockMove(const BlockMove& move){
        processRecursiveFunction( [&]() {
            m_blockCountPriorPtr->applyBlockMove(move);
            applyBlockMoveToVertexCounts(move);
            applyBlockMoveToState(move);
        });
    }

    void computationFinished() override {
        m_isProcessed=false;
        m_blockCountPriorPtr->computationFinished();
    }

    void checkSelfConsistency() const {
        checkBlockSequenceConsistencyWithBlockCount(m_state, getBlockCount());
        checkBlockSequenceConsistencyWithVertexCountsInBlocks(m_state, getVertexCountsInBlocks());
    };

};

class BlockHyperPrior: public BlockPrior{
protected:
    VertexCountPrior* m_vertexCountPriorPtr = NULL;
public:
    BlockHyperPrior() {}
    BlockHyperPrior(VertexCountPrior& vertexCountPrior):
        m_vertexCountPriorPtr(&vertexCountPrior),
        BlockPrior(vertexCountPrior.getSize()){ m_vertexCountPriorPtr->isRoot(false); }
    BlockHyperPrior(const BlockHyperPrior& other){
        this->setState(other.m_state);
        this->setVertexCountPrior(*other.m_vertexCountPriorPtr);
    }
    virtual ~BlockHyperPrior(){}
    const BlockHyperPrior& operator=(const BlockHyperPrior& other){
        this->setState(other.m_state);
        this->setVertexCountPrior(*other.m_vertexCountPriorPtr);
        return *this;
    }

    const VertexCountPrior& getVertexCountPrior() const{ return *m_vertexCountPriorPtr; }
    VertexCountPrior& getVertexCountPriorRef() const{ return *m_vertexCountPriorPtr; }
    void setVertexCountPrior(VertexCountPrior& vertexCountPrior){
        m_vertexCountPriorPtr = &vertexCountPrior;
    }

    void setState(const BlockSequence& blockSeq) override{
        BlockPrior::setState(blockSeq);
        m_vertexCountPriorPtr->setState( m_vertexCountsInBlocks );
    }
    const std::vector<size_t>& getVertexCountsInBlocks() const {
        return m_vertexCountPriorPtr->getState();
    }
    void sampleState();

    void samplePriors() override {
        m_vertexCountPriorPtr->sample();
    }

    double getLogLikelihood() const ;
    double getLogPrior() { return m_vertexCountPriorPtr->getLogJoint(); }

    double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const;
    double getLogPriorRatioFromBlockMove(const BlockMove& move) {
        return m_vertexCountPriorPtr->getLogJointRatioFromBlockMove(move);
    };

    void applyBlockMove(const BlockMove& move){
        processRecursiveFunction( [&]() {
            m_vertexCountPriorPtr->applyBlockMove(move);
            applyBlockMoveToState(move);
        });
    }
    void computationFinished() override {
        m_isProcessed=false;
        m_vertexCountPriorPtr->computationFinished();
    }

    void checkSelfConsistency() const {
        m_vertexCountPriorPtr->checkSelfConsistency();
        checkBlockSequenceConsistencyWithVertexCountsInBlocks(m_state, getVertexCountsInBlocks());
    };
};

class BlockUniformHyperPrior: public BlockHyperPrior{
public:
    BlockUniformHyperPrior(){}
    BlockUniformHyperPrior(size_t size, BlockCountPrior& blockCountPrior):
        BlockHyperPrior(*new VertexCountUniformPrior(size, blockCountPrior)){}
    BlockUniformHyperPrior(const BlockUniformHyperPrior& other):
        BlockHyperPrior(*other.m_vertexCountPriorPtr){ }
    virtual ~BlockUniformHyperPrior(){ delete m_vertexCountPriorPtr; }
    const BlockUniformHyperPrior& operator=(const BlockUniformHyperPrior& other){
        this->setState(other.m_state);
        this->setVertexCountPrior(*other.m_vertexCountPriorPtr);
        return *this;
    }
};

}

#endif
