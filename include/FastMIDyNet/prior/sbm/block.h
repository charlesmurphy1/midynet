#ifndef FAST_MIDYNET_BLOCK_H
#define FAST_MIDYNET_BLOCK_H

#include <vector>
#include <iostream>

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/layer_count.h"
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
    BlockPrior(size_t size):
        m_size(size){ }

    virtual void setState(const BlockSequence& blocks) override{
        m_blockCount = computeBlockCount(blocks);
        m_vertexCountsInBlocks = computeVertexCountsInBlocks(blocks);
        m_state = blocks;
    }

    virtual const size_t& getBlockCount() const { return m_blockCount; }
    virtual const std::vector<size_t>& getVertexCountsInBlocks() const { return m_vertexCountsInBlocks; };
    const BlockIndex& getBlockOfIdx(BaseGraph::VertexIndex idx) const { return m_state[idx]; }
    static size_t computeBlockCount(const BlockSequence& blocks) { return *max_element(blocks.begin(), blocks.end()) + 1; }
    static std::vector<size_t> computeVertexCountsInBlocks(const BlockSequence&);
    const size_t& getSize() const { return m_size; }

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
        else if (move.addedBlocks == -1) m_vertexCountsInBlocks.pop_back();

        --m_vertexCountsInBlocks[move.prevBlockIdx];
        ++m_vertexCountsInBlocks[move.nextBlockIdx];
    };
    virtual void applyBlockMove(const BlockMove&) = 0;
    void computationFinished() override { m_isProcessed=false; }

    static void checkBlockSequenceConsistencyWithBlockCount(const BlockSequence& blockSeq, size_t expectedBlockCount) ;
    static void checkBlockSequenceConsistencyWithVertexCountsInBlocks(const BlockSequence& blockSeq, std::vector<size_t> expectedVertexCountsInBlocks) ;

};

class BlockDeltaPrior: public BlockPrior{
    BlockSequence m_blockSeq;
public:
    BlockDeltaPrior(const BlockSequence& blockSeq):
        m_blockSeq(blockSeq),
        BlockPrior(blockSeq.size()) { setState(blockSeq); }

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
    BlockCountPrior& m_blockCountPrior;
public:
    BlockUniformPrior(size_t graphSize, BlockCountPrior& blockCountPrior):
        BlockPrior(graphSize), m_blockCountPrior(blockCountPrior) { }

    const size_t& getBlockCount() const { return m_blockCountPrior.getState();}
    void setState(const BlockSequence& blockSeq) override{
        m_state = blockSeq;
        m_blockCountPrior.setState(BlockPrior::getBlockCount());
    }
    void sampleState();
    void samplePriors() { m_blockCountPrior.sample(); }

    double getLogLikelihood() const ;
    double getLogPrior() { return m_blockCountPrior.getLogJoint(); };

    double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const;
    double getLogPriorRatioFromBlockMove(const BlockMove& move) {
        return m_blockCountPrior.getLogJointRatioFromBlockMove(move);
    };

    void applyBlockMove(const BlockMove& move){
        processRecursiveFunction( [&]() {
            m_blockCountPrior.applyBlockMove(move);
            applyBlockMoveToVertexCounts(move);
            applyBlockMoveToState(move);
        });
    }

    void computationFinished() override {
        m_isProcessed=false;
        m_blockCountPrior.computationFinished();
    }

    void checkSelfConsistency() const {
        checkBlockSequenceConsistencyWithBlockCount(m_state, getBlockCount());
        checkBlockSequenceConsistencyWithVertexCountsInBlocks(m_state, getVertexCountsInBlocks());
    };

};

class BlockHyperPrior: public BlockPrior{
public:
    BlockHyperPrior(VertexCountPrior& vertexCountPrior):
        m_vertexCountPrior(vertexCountPrior),
        BlockPrior(vertexCountPrior.getSize()){ }

    void setState(const BlockSequence& blockSeq) override{
        m_vertexCountPrior.setState( computeVertexCountsInBlocks(blockSeq) );
        m_state = blockSeq;
    }
    const std::vector<size_t>& getVertexCountsInBlocks() const { return m_vertexCountPrior.getState(); };
    void sampleState();

    void samplePriors() override {
        m_vertexCountPrior.sample();
    }

    double getLogLikelihood() const ;
    double getLogPrior() { return m_vertexCountPrior.getLogJoint(); }

    double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const;
    double getLogPriorRatioFromBlockMove(const BlockMove& move) {
        return m_vertexCountPrior.getLogJointRatioFromBlockMove(move);
    };

    void applyBlockMove(const BlockMove& move){
        processRecursiveFunction( [&]() {
            m_vertexCountPrior.applyBlockMove(move);
            applyBlockMoveToState(move);
        });
    }
    void computationFinished() override {
        m_isProcessed=false;
        m_vertexCountPrior.computationFinished();
    }


    void checkSelfConsistency() const {
        m_vertexCountPrior.checkSelfConsistency();
        checkBlockSequenceConsistencyWithVertexCountsInBlocks(m_state, getVertexCountsInBlocks());
    };
protected:
    VertexCountPrior& m_vertexCountPrior;
};

class BlockUniformHyperPrior: public BlockHyperPrior{
public:
    BlockUniformHyperPrior(size_t size, BlockCountPrior& blockCountPrior):
        BlockHyperPrior(*new VertexCountUniformPrior(size, blockCountPrior)){}
    BlockUniformHyperPrior(const BlockUniformHyperPrior& other):
        BlockHyperPrior(other.m_vertexCountPrior){ }
    ~BlockUniformHyperPrior(){ delete & m_vertexCountPrior; }


};

class BlockHierarchicalPrior: public BlockPrior{
private:
    LayerCountPrior& m_layerCountPrior;
    std::vector<BlockSequence> m_hierarchicalState;
public:
    BlockHierarchicalPrior(size_t size, LayerCountPrior& layerCountPrior):
    BlockPrior(size), m_layerCountPrior(layerCountPrior){}
};

class BlockHierarchicalUniformPrior: public BlockHierarchicalPrior{

};

class BlockHierarchicalHyperPrior: public BlockHierarchicalPrior{

};


}

#endif
