#ifndef FAST_MIDYNET_SBM_H
#define FAST_MIDYNET_SBM_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "prior/label_graph.h"
#include "prior/block.h"
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
    LabelGraphPrior* m_labelGraphPriorPtr = nullptr;
protected:

    void _applyGraphMove (const GraphMove& move) override {
        m_labelGraphPriorPtr->applyGraphMove(move);
        RandomGraph::_applyGraphMove(move);
    }
    void _applyLabelMove (const BlockMove& move) override {
        m_labelGraphPriorPtr->applyLabelMove(move);
    }
    const double _getLogPrior() const override { return m_labelGraphPriorPtr->getLogJoint(); }
    const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const override { return m_labelGraphPriorPtr->getLogJointRatioFromGraphMove(move); }
    const double _getLogPriorRatioFromLabelMove(const BlockMove& move) const override {
        return m_labelGraphPriorPtr->getLogJointRatioFromLabelMove(move);
    }
    void _samplePrior() override { m_labelGraphPriorPtr->sample(); }
    void setUpLikelihood() override {
        m_likelihoodModel.m_graphPtr = &m_graph;
        m_likelihoodModel.m_labelGraphPriorPtrPtr = &m_labelGraphPriorPtr;
    }

public:
    StochasticBlockModelFamily(size_t graphSize): VertexLabeledRandomGraph<BlockIndex>(graphSize, m_likelihoodModel) {
        setUpLikelihood();
    }
    StochasticBlockModelFamily(size_t graphSize, BlockPrior& blockPrior, LabelGraphPrior& labelGraphPrior):
        VertexLabeledRandomGraph<BlockIndex>(graphSize, m_likelihoodModel), m_labelGraphPriorPtr(&labelGraphPrior){
            setUpLikelihood();
            setBlockPrior(blockPrior);
        }

    void sampleState () override;
    void sampleLabels() override {
        m_labelGraphPriorPtr->samplePartition();
    }

    void setGraph(const MultiGraph graph) override{
        RandomGraph::setGraph(graph);
        m_labelGraphPriorPtr->setGraph(m_graph);
    }
    void setLabels(const std::vector<BlockIndex>& labels) override { m_labelGraphPriorPtr->setPartition(labels); }


    const BlockPrior& getBlockPrior() const { return m_labelGraphPriorPtr->getBlockPrior(); }
    void setBlockPrior(BlockPrior& blockPrior) {
        if (m_labelGraphPriorPtr == nullptr)
            throw SafetyError("StochasticBlockModelFamily", "m_labelGraphPriorPtr");
        m_labelGraphPriorPtr->setBlockPrior(blockPrior);
    }

    const LabelGraphPrior& getLabelGraphPrior() const { return *m_labelGraphPriorPtr; }
    void setLabelGraphPrior(LabelGraphPrior& labelGraphPrior) {
        m_labelGraphPriorPtr = &labelGraphPrior;
        m_labelGraphPriorPtr->isRoot(false);
    }

    const BlockSequence& getLabels() const override { return m_labelGraphPriorPtr->getBlockPrior().getState(); }
    const size_t getLabelCount() const override { return m_labelGraphPriorPtr->getBlockPrior().getBlockCount(); }
    const CounterMap<BlockIndex>& getVertexCounts() const override { return m_labelGraphPriorPtr->getBlockPrior().getVertexCounts(); }
    const CounterMap<BlockIndex>& getEdgeLabelCounts() const override { return m_labelGraphPriorPtr->getEdgeCounts(); }
    const LabelGraph& getLabelGraph() const override { return m_labelGraphPriorPtr->getState(); }
    const size_t getEdgeCount() const override { return m_labelGraphPriorPtr->getEdgeCount(); }

    virtual void checkSelfConsistency() const override;
    virtual const bool isCompatible(const MultiGraph& graph) const override{
        if (not VertexLabeledRandomGraph<BlockIndex>::isCompatible(graph)) return false;
        auto labelGraph = getLabelGraphFromGraph(graph, getLabels());
        return labelGraph.getAdjacencyMatrix() == getLabelGraph().getAdjacencyMatrix();
    }
    void computationFinished() const override {
        m_isProcessed = false;
        m_labelGraphPriorPtr->computationFinished();
    }
    void checkSelfSafety() const override {
        RandomGraph::checkSelfSafety();
        if (not m_labelGraphPriorPtr)
            throw SafetyError("StochasticBlockModelFamily", "m_labelGraphPriorPtr");
    }
};


class UniformStochasticBlockModel: public StochasticBlockModelFamily{

};

class HyperUniformStochasticBlockModel: public StochasticBlockModelFamily{

};

}// end FastMIDyNet
#endif
