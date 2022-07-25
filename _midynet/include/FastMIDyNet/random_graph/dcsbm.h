#ifndef FAST_MIDYNET_DCSBM_H
#define FAST_MIDYNET_DCSBM_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "prior/block.h"
#include "prior/label_graph.h"
#include "prior/labeled_degree.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/random_graph/util.h"
#include "FastMIDyNet/random_graph/likelihood/dcsbm.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class DegreeCorrectedStochasticBlockModelFamily: public BlockLabeledRandomGraph{
private:
    DegreeCorrectedStochasticBlockModelLikelihood m_likelihoodModel = DegreeCorrectedStochasticBlockModelLikelihood();
    VertexLabeledDegreePrior* m_degreePriorPtr = nullptr;
protected:
    void _applyGraphMove (const GraphMove& move) override {
        RandomGraph::_applyGraphMove(move);
        m_degreePriorPtr->applyGraphMove(move);
    }
    void _applyLabelMove (const BlockMove& move) override{
        m_degreePriorPtr->applyLabelMove(move);
    }
    const double _getLogPrior() const override { return m_degreePriorPtr->getLogJoint(); }
    const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const override { return m_degreePriorPtr->getLogJointRatioFromGraphMove(move); }
    const double _getLogPriorRatioFromLabelMove(const BlockMove& move) const override { return m_degreePriorPtr->getLogJointRatioFromLabelMove(move); }
    void _samplePrior() override { m_degreePriorPtr->sample(); }
    void setUpLikelihood() override {
        m_likelihoodModel.m_graphPtr = &m_graph;
        m_likelihoodModel.m_degreePriorPtrPtr = &m_degreePriorPtr;
    }

public:
    DegreeCorrectedStochasticBlockModelFamily(size_t graphSize):
        VertexLabeledRandomGraph<BlockIndex>(graphSize, m_likelihoodModel) { setUpLikelihood(); }
    DegreeCorrectedStochasticBlockModelFamily(size_t graphSize, BlockPrior& blockPrior, LabelGraphPrior& labelGraphPrior, VertexLabeledDegreePrior& degreePrior):
        VertexLabeledRandomGraph<BlockIndex>(graphSize, m_likelihoodModel), m_degreePriorPtr(&degreePrior){
            setUpLikelihood();
            m_degreePriorPtr->isRoot(false);
            setBlockPrior(blockPrior);
            setLabelGraphPrior(labelGraphPrior);
        }

    void sampleState () override;
    void sampleLabels() override {
        m_degreePriorPtr->samplePartition();
    }

    void setGraph(const MultiGraph graph) override {
        RandomGraph::setGraph(graph);
        m_degreePriorPtr->setGraph(m_graph);
    }
    void setLabels(const std::vector<BlockIndex>& labels) override {
        m_degreePriorPtr->setPartition(labels);
    }


    const BlockPrior& getBlockPrior() const { return m_degreePriorPtr->getBlockPrior(); }
    void setBlockPrior(BlockPrior& prior) {
        if (m_degreePriorPtr == nullptr)
            throw SafetyError("DegreeCorrectedStochasticBlockModelFamily: unsafe degree prior with value `nullptr`.");
        m_degreePriorPtr->setBlockPrior(prior);
    }

    const LabelGraphPrior& getLabelGraphPrior() const { return m_degreePriorPtr->getLabelGraphPrior(); }
    void setLabelGraphPrior(LabelGraphPrior& prior) {
        if (m_degreePriorPtr == nullptr)
            throw SafetyError("DegreeCorrectedStochasticBlockModelFamily: unsafe degree prior with value `nullptr`.");
        m_degreePriorPtr->setLabelGraphPrior(prior);
    }

    const VertexLabeledDegreePrior& getDegreePrior() const { return *m_degreePriorPtr; }
    void setDegreePrior(VertexLabeledDegreePrior& prior) {
        prior.isRoot(false);
        m_degreePriorPtr = &prior;
    }

    const BlockSequence& getLabels() const override { return getBlockPrior().getState(); }
    const size_t getLabelCount() const override { return getBlockPrior().getBlockCount(); }
    const CounterMap<BlockIndex>& getLabelCounts() const override { return getBlockPrior().getVertexCounts(); }
    const CounterMap<BlockIndex>& getEdgeLabelCounts() const override { return getLabelGraphPrior().getEdgeCounts(); }
    const LabelGraph& getLabelGraph() const override { return getLabelGraphPrior().getState(); }
    const size_t& getEdgeCount() const override { return getLabelGraphPrior().getEdgeCount(); }
    const std::vector<size_t> getDegrees() const { return getDegreePrior().getState(); }

    void checkSelfConsistency() const override;
    const bool isCompatible(const MultiGraph& graph) const override{
        if (not VertexLabeledRandomGraph<BlockIndex>::isCompatible(graph)) return false;
        auto labelGraph = getLabelGraphFromGraph(graph, getLabels());
        bool sameLabelGraph = labelGraph.getAdjacencyMatrix() == getLabelGraph().getAdjacencyMatrix() ;
        bool sameDegrees = graph.getDegrees() == getDegrees();
        return sameLabelGraph and sameDegrees;
    }
    void computationFinished() const override {
        m_isProcessed = false;
        m_degreePriorPtr->computationFinished();
    }
};

}// end FastMIDyNet
#endif
