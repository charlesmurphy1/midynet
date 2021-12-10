#ifndef FAST_MIDYNET_HIERARCHICAL_BLOCK_H
#define FAST_MIDYNET_HIERARCHICAL_BLOCK_H

#include <vector>
#include <iostream>

#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/layer_count.h"
#include "FastMIDyNet/prior/sbm/vertex_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class BlockHierarchicalPrior: public Prior<NestedBlockSequence>{
protected:
    size_t m_size;
    size_t m_layerCount;
    std::vector<size_t> m_blockCounts;
    std::vector<std::vector<size_t>> m_vertexCountsInBlocks;
public:
    BlockHierarchicalPrior(size_t size):
        m_size(size){ }

    virtual void setState(const NestedBlockSequence& nestedBlocks) override{
        m_blockCounts.clear();
        m_vertexCountsInBlocks.clear();
        for (auto b : nestedBlocks) {
            m_blockCounts.push_back(computeBlockCount(b));
            m_vertexCountsInBlocks.push_back(computeVertexCountsInBlocks(b));
        }
        m_state = nestedBlocks;
    }

    virtual const size_t& getLayerCount() const { return m_layerCount; }
    virtual const std::vector<size_t>& getBlockCounts() const { return m_blockCounts; }
    virtual const std::vector<std::vector<size_t>>& getVertexCountsInBlocks() const { return m_vertexCountsInBlocks; };
    const BlockIndex& getBlockOfIdxInLayer(size_t layerIdx, BaseGraph::VertexIndex idx) const { return m_state[layerIdx][idx]; }
    const std::vector<BlockIndex> getBlockHierarchyOfIdx(BaseGraph::VertexIndex idx) const ;
    const size_t& getSize() const { return m_size; }

    double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const { return 0; };
    virtual double getLogLikelihoodRatioFromNestedBlockMove(const NestedBlockMove& move) const = 0;

    double getLogPriorRatioFromGraphMove(const GraphMove& move) { return 0; };
    virtual double getLogPriorRatioFromNestedBlockMove(const NestedBlockMove& move) = 0;

    double getLogJointRatioFromGraphMove(const GraphMove& move) { return 0; };
    double getLogJointRatioFromNestedBlockMove(const NestedBlockMove& move) {
        return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromNestedBlockMove(move) + getLogPriorRatioFromNestedBlockMove(move); }, 0);
    };

    void applyGraphMove(const GraphMove&) { };
    void applyBlockMoveToState(const NestedBlockMove& move) {
        for(size_t l=0; l<getLayerCount(); ++l){
            auto m = move[l];
             m_state[l][m.vertexIdx] = m.nextBlockIdx;
        }
    };
    void applyBlockMoveToVertexCounts(const NestedBlockMove& move) ;

    virtual void applyNestedBlockMove(const NestedBlockMove&) = 0;
    void computationFinished() override { m_isProcessed=false; }

    static void checkBlockSequenceConsistencyWithBlockCount(const BlockSequence&, size_t) ;
    static void checkBlockSequenceConsistencyWithVertexCountsInBlocks(const BlockSequence&, std::vector<size_t>) ;
    static size_t computeBlockCount(const BlockSequence& blocks) { return *max_element(blocks.begin(), blocks.end()) + 1; }
    static std::vector<size_t> computeVertexCountsInBlocks(const BlockSequence&);
};



// class BlockHierarchicalUniformPrior: public BlockHierarchicalPrior{
// private:
//     NestedBlockCountPrior& m_nestedBlockCountPrior;
// public:
//     BlockHierarchicalUniformPrior(size_t graphSize, LayerCountPrior& layerCountPrior):
//         BlockHierarchicalPrior(graphSize), m_layerCountPrior(layerCountPrior) {
//         }
//
//     const size_t& getLayerCount() const { return m_layerCountPrior.getState();}
//     void setState(const NestedBlockSequence& blockSeq) override{
//         BlockHierarchicalPrior::setState(blockSeq);
//         m_layerCountPrior.setState(m_layerCount);
//     }
//     void sampleState();
//     void samplePriors() { m_layerCountPrior.sample(); }
//
//     double getLogLikelihood() const ;
//     double getLogPrior() { return m_blockCountPrior.getLogJoint(); };
//
//     double getLogLikelihoodRatioFromNestedBlockMove(const NestedBlockMove&) const;
//     double getLogPriorRatioFromNestedBlockMove(const NestedBlockMove& move) {
//         return m_blockCountPrior.getLogJointRatioFromNestedBlockMove(move);
//     };
//
//     void applyNestedBlockMove(const NestedBlockMove& move){
//         processRecursiveFunction( [&]() {
//             m_blockCountPrior.applyNestedBlockMove(move);
//             applyBlockMoveToVertexCounts(move);
//             applyBlockMoveToState(move);
//         });
//     }
//
//     void computationFinished() override {
//         m_isProcessed=false;
//         m_blockCountPrior.computationFinished();
//     }
//
//     void checkSelfConsistency() const {
//         checkBlockSequenceConsistencyWithBlockCount(m_state, getBlockCount());
//         checkBlockSequenceConsistencyWithVertexCountsInBlocks(m_state, getVertexCountsInBlocks());
//     };
// };


}

#endif
