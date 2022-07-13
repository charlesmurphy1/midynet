#ifndef FAST_MIDYNET_SBM_H
#define FAST_MIDYNET_SBM_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/random_graph/likelihood.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class StochasticBlockModelFamily: public BlockLabeledRandomGraph, public StochasticBlockModelLikelihood{
protected:
    BlockPrior* m_blockPriorPtr = nullptr;
    EdgeMatrixPrior* m_edgeMatrixPriorPtr = nullptr;

    virtual void _applyGraphMove (const GraphMove& move) override {
        m_blockPriorPtr->applyGraphMove(move);
        m_edgeMatrixPriorPtr->applyGraphMove(move);
        RandomGraph::_applyGraphMove(move);
    }
    virtual void _applyLabelMove (const BlockMove& move) override{
        m_blockPriorPtr->applyLabelMove(move);
        m_edgeMatrixPriorPtr->applyLabelMove(move);
    }

public:
    StochasticBlockModelFamily(size_t graphSize):
        VertexLabeledRandomGraph<BlockIndex>(graphSize), StochasticBlockModelLikelihood() {
            m_sizePtr = &m_size;
            m_graphPtr = &m_graph;
            m_blockPriorPtrPtr = &m_blockPriorPtr;
            m_edgeMatrixPriorPtrPtr = &m_edgeMatrixPriorPtr;
    }
    StochasticBlockModelFamily(size_t graphSize, BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior):
        VertexLabeledRandomGraph<BlockIndex>(graphSize){
            m_sizePtr = &m_size;
            m_graphPtr = &m_graph;
            m_blockPriorPtrPtr = &m_blockPriorPtr;
            m_edgeMatrixPriorPtrPtr = &m_edgeMatrixPriorPtr;

            setBlockPrior(blockPrior);
            m_blockPriorPtr->setSize(graphSize);
            setEdgeMatrixPrior(edgeMatrixPrior);
        }

    void sample () override;
    void sampleLabels() override {
        m_blockPriorPtr->sample();
        setLabels(m_blockPriorPtr->getState());
    }

    void setGraph(const MultiGraph& graph) override{
        RandomGraph::setGraph(graph);
        m_edgeMatrixPriorPtr->setGraph(m_graph);
    }
    
    void setLabels(const std::vector<BlockIndex>& labels) override {
        m_edgeMatrixPriorPtr->setPartition(labels);
    }


    const BlockPrior& getBlockPrior() const { return *m_blockPriorPtr; }
    BlockPrior& getBlockPriorRef() const { return *m_blockPriorPtr; }
    virtual void setBlockPrior(BlockPrior& blockPrior) {
        m_blockPriorPtr = &blockPrior;
        m_blockPriorPtr->isRoot(false);
        m_blockPriorPtr->setSize(m_size);
        if (m_edgeMatrixPriorPtr)
            m_edgeMatrixPriorPtr->setBlockPrior(*m_blockPriorPtr);
    }

    const EdgeMatrixPrior& getEdgeMatrixPrior() const { return *m_edgeMatrixPriorPtr; }
    EdgeMatrixPrior& getEdgeMatrixPriorRef() const { return *m_edgeMatrixPriorPtr; }
    virtual void setEdgeMatrixPrior(EdgeMatrixPrior& edgeMatrixPrior) {
        m_edgeMatrixPriorPtr = &edgeMatrixPrior;
        m_edgeMatrixPriorPtr->isRoot(false);
        m_edgeMatrixPriorPtr->setBlockPrior(*m_blockPriorPtr);
    }

    const BlockSequence& getLabels() const override { return m_blockPriorPtr->getState(); }
    const size_t getLabelCount() const override { return m_blockPriorPtr->getBlockCount(); }
    const CounterMap<BlockIndex>& getLabelCounts() const override { return m_blockPriorPtr->getVertexCounts(); }
    const CounterMap<BlockIndex>& getEdgeLabelCounts() const override { return m_edgeMatrixPriorPtr->getEdgeCounts(); }
    const MultiGraph& getLabelGraph() const override { return m_edgeMatrixPriorPtr->getState(); }
    const size_t& getEdgeCount() const override { return m_edgeMatrixPriorPtr->getEdgeCount(); }

    virtual const double getLogLikelihood() const override { return StochasticBlockModelLikelihood::getLogLikelihood(); }
    virtual const double getLogPrior() const override {
        return processRecursiveConstFunction<double>([&](){
            return m_blockPriorPtr->getLogJoint() + m_edgeMatrixPriorPtr->getLogJoint();
        }, 0);
    }

    virtual const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const override {
        return StochasticBlockModelLikelihood::getLogLikelihoodRatioFromGraphMove(move);
    }
    virtual const double getLogLikelihoodRatioFromLabelMove (const BlockMove& move) const override {
        return StochasticBlockModelLikelihood::getLogLikelihoodRatioFromLabelMove(move);
    }

    virtual const double getLogPriorRatioFromGraphMove (const GraphMove& move) const override {
        return processRecursiveConstFunction<double>([&](){
            return m_blockPriorPtr->getLogJointRatioFromGraphMove(move) + m_edgeMatrixPriorPtr->getLogJointRatioFromGraphMove(move);
        }, 0);
    }
    virtual const double getLogPriorRatioFromLabelMove (const BlockMove& move) const override {
        return processRecursiveConstFunction<double>([&](){
            return m_blockPriorPtr->getLogJointRatioFromLabelMove(move) + m_edgeMatrixPriorPtr->getLogJointRatioFromLabelMove(move);
        }, 0);
    }


    virtual bool isSafe() const override { return m_blockPriorPtr != nullptr and m_edgeMatrixPriorPtr != nullptr; }

    static MultiGraph getEdgeMatrixFromGraph(const MultiGraph&, const BlockSequence&) ;
    static void checkGraphConsistencyWithEdgeMatrix(const MultiGraph& graph, const BlockSequence& blockSeq, const MultiGraph& expectedEdgeMat);
    virtual void checkSelfConsistency() const override;
    virtual void checkSelfSafety() const override;
    virtual const bool isCompatible(const MultiGraph& graph) const override{
        if (not VertexLabeledRandomGraph<BlockIndex>::isCompatible(graph)) return false;
        auto edgeMatrix = getEdgeMatrixFromGraph(graph, getLabels());
        return edgeMatrix.getAdjacencyMatrix() == m_edgeMatrixPriorPtr->getState().getAdjacencyMatrix();
    };
    virtual void computationFinished() const override {
        m_isProcessed = false;
        m_blockPriorPtr->computationFinished();
        m_edgeMatrixPriorPtr->computationFinished();
    }
};

}// end FastMIDyNet
#endif
