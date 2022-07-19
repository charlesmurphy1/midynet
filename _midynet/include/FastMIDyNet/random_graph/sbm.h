#ifndef FAST_MIDYNET_SBM_H
#define FAST_MIDYNET_SBM_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/random_graph/util.h"
#include "FastMIDyNet/random_graph/likelihood/sbm.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class StochasticBlockModelFamily: public BlockLabeledRandomGraph{
private:
    StochasticBlockModelLikelihood m_likelihoodModel;
    EdgeMatrixPrior* m_edgeMatrixPriorPtr = nullptr;
protected:

    void _applyGraphMove (const GraphMove& move) override {
        m_edgeMatrixPriorPtr->applyGraphMove(move);
        RandomGraph::_applyGraphMove(move);
    }
    void _applyLabelMove (const BlockMove& move) override {
        m_edgeMatrixPriorPtr->applyLabelMove(move);
    }
    const double _getLogPrior() const override { return m_edgeMatrixPriorPtr->getLogJoint(); }
    const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const override { return m_edgeMatrixPriorPtr->getLogJointRatioFromGraphMove(move); }
    const double _getLogPriorRatioFromLabelMove(const BlockMove& move) const override {
        return m_edgeMatrixPriorPtr->getLogJointRatioFromLabelMove(move);
    }
    void _samplePrior() override { m_edgeMatrixPriorPtr->sample(); }
    void setUpLikelihood() override {
        m_likelihoodModel.m_graphPtr = &m_graph;
        m_likelihoodModel.m_edgeMatrixPriorPtrPtr = &m_edgeMatrixPriorPtr;
    }

public:
    StochasticBlockModelFamily(size_t graphSize): VertexLabeledRandomGraph<BlockIndex>(graphSize, m_likelihoodModel) {
        setUpLikelihood();
    }
    StochasticBlockModelFamily(size_t graphSize, BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior):
        VertexLabeledRandomGraph<BlockIndex>(graphSize, m_likelihoodModel), m_edgeMatrixPriorPtr(&edgeMatrixPrior){
            setUpLikelihood();
            setBlockPrior(blockPrior);
        }

    void sampleState () override;
    void sampleLabels() override {
        m_edgeMatrixPriorPtr->samplePartition();
    }

    void setGraph(const MultiGraph graph) override{
        RandomGraph::setGraph(graph);
        m_edgeMatrixPriorPtr->setGraph(m_graph);
    }
    void setLabels(const std::vector<BlockIndex>& labels) override { m_edgeMatrixPriorPtr->setPartition(labels); }


    const BlockPrior& getBlockPrior() const { return m_edgeMatrixPriorPtr->getBlockPrior(); }
    void setBlockPrior(BlockPrior& blockPrior) {
        if (m_edgeMatrixPriorPtr == nullptr)
            throw SafetyError("StochasticBlockModelFamily: unsafe edge matrix prior with value `nullptr`.");
        m_edgeMatrixPriorPtr->setBlockPrior(blockPrior);
    }

    const EdgeMatrixPrior& getEdgeMatrixPrior() const { return *m_edgeMatrixPriorPtr; }
    void setEdgeMatrixPrior(EdgeMatrixPrior& edgeMatrixPrior) {
        m_edgeMatrixPriorPtr = &edgeMatrixPrior;
        m_edgeMatrixPriorPtr->isRoot(false);
    }

    const BlockSequence& getLabels() const override { return m_edgeMatrixPriorPtr->getBlockPrior().getState(); }
    const size_t getLabelCount() const override { return m_edgeMatrixPriorPtr->getBlockPrior().getBlockCount(); }
    const CounterMap<BlockIndex>& getLabelCounts() const override { return m_edgeMatrixPriorPtr->getBlockPrior().getVertexCounts(); }
    const CounterMap<BlockIndex>& getEdgeLabelCounts() const override { return m_edgeMatrixPriorPtr->getEdgeCounts(); }
    const MultiGraph& getLabelGraph() const override { return m_edgeMatrixPriorPtr->getState(); }
    const size_t& getEdgeCount() const override { return m_edgeMatrixPriorPtr->getEdgeCount(); }

    virtual void checkSelfConsistency() const override;
    virtual const bool isCompatible(const MultiGraph& graph) const override{
        if (not VertexLabeledRandomGraph<BlockIndex>::isCompatible(graph)) return false;
        auto edgeMatrix = getEdgeMatrixFromGraph(graph, getLabels());
        return edgeMatrix.getAdjacencyMatrix() == getLabelGraph().getAdjacencyMatrix();
    }
    void computationFinished() const override {
        m_isProcessed = false;
        m_edgeMatrixPriorPtr->computationFinished();
    }
};

}// end FastMIDyNet
#endif
