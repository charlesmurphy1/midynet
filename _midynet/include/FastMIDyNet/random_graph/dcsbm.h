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

class DegreeCorrectedStochasticBlockModelBase: public BlockLabeledRandomGraph{
protected:
    DegreeCorrectedStochasticBlockModelLikelihood m_likelihoodModel = DegreeCorrectedStochasticBlockModelLikelihood();
    VertexLabeledDegreePrior* m_degreePriorPtr = nullptr;
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
        m_likelihoodModel.m_statePtr = &m_state;
        m_likelihoodModel.m_degreePriorPtrPtr = &m_degreePriorPtr;
    }

public:
    DegreeCorrectedStochasticBlockModelBase(size_t graphSize):
        VertexLabeledRandomGraph<BlockIndex>(graphSize, m_likelihoodModel) { setUpLikelihood(); }
    DegreeCorrectedStochasticBlockModelBase(size_t graphSize, VertexLabeledDegreePrior& degreePrior):
        VertexLabeledRandomGraph<BlockIndex>(graphSize, m_likelihoodModel), m_degreePriorPtr(&degreePrior){
            setUpLikelihood();
            m_degreePriorPtr->isRoot(false);
        }

    void sampleLabels() override {
        m_degreePriorPtr->samplePartition();
    }

    void setState(const MultiGraph& state) override {
        RandomGraph::setState(state);
        m_degreePriorPtr->setGraph(m_state);
    }
    void setLabels(const std::vector<BlockIndex>& labels) override {
        m_degreePriorPtr->setPartition(labels);
    }


    const VertexLabeledDegreePrior& getDegreePrior() const { return *m_degreePriorPtr; }
    VertexLabeledDegreePrior& getDegreePriorRef() const { return *m_degreePriorPtr; }
    void setDegreePrior(VertexLabeledDegreePrior& prior) {
        prior.isRoot(false);
        m_degreePriorPtr = &prior;
    }

    const BlockSequence& getLabels() const override {
        return m_degreePriorPtr->getLabelGraphPrior().getBlockPrior().getState();
    }
    const size_t getLabelCount() const override {
        return m_degreePriorPtr->getLabelGraphPrior().getBlockPrior().getBlockCount();
    }
    const CounterMap<BlockIndex>& getVertexCounts() const override {
        return m_degreePriorPtr->getLabelGraphPrior().getBlockPrior().getVertexCounts();
    }
    const CounterMap<BlockIndex>& getEdgeLabelCounts() const override {
        return m_degreePriorPtr->getLabelGraphPrior().getEdgeCounts();
    }
    const LabelGraph& getLabelGraph() const override {
        return m_degreePriorPtr->getLabelGraphPrior().getState();
    }
    const size_t getEdgeCount() const override {
        return m_degreePriorPtr->getLabelGraphPrior().getEdgeCount();
    }
    const std::vector<size_t> getDegrees() const { return getDegreePrior().getState(); }

    virtual void checkSelfConsistency() const override {
        m_degreePriorPtr->checkSelfConsistency();
        checkGraphConsistencyWithLabelGraph("DegreeCorrectedStochasticBlockModelBase", m_state, getLabels(), getLabelGraph());
        checkGraphConsistencyWithDegreeSequence("DegreeCorrectedStochasticBlockModelBase", m_state, getDegrees());
    }
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

    void checkSelfSafety() const override {
        RandomGraph::checkSelfSafety();
        if (not m_degreePriorPtr)
            throw SafetyError("DegreeCorrectedStochasticBlockModelBase", "m_degreePriorPtr");
        m_degreePriorPtr->checkSafety();
    }
};

class DegreeCorrectedStochasticBlockModelFamily: public DegreeCorrectedStochasticBlockModelBase{
    BlockCountUniformPrior m_blockCountPrior;
    LabelGraphErdosRenyiPrior m_labelGraphPrior;

    BlockPrior* m_blockPriorPtr;
    EdgeCountPrior* m_edgeCountPriorPtr;
public:
    DegreeCorrectedStochasticBlockModelFamily(size_t size, double edgeCount, bool useHyperPrior=false, bool canonical=false):
        DegreeCorrectedStochasticBlockModelBase(size),
        m_blockCountPrior(1, size-1),
        m_labelGraphPrior(){
            m_edgeCountPriorPtr = makeEdgeCountPrior(edgeCount, canonical);
            m_blockPriorPtr = makeBlockPrior(size, m_blockCountPrior, useHyperPrior);
            m_labelGraphPrior = LabelGraphErdosRenyiPrior(*m_edgeCountPriorPtr, *m_blockPriorPtr);
            m_degreePriorPtr = makeVertexLabeledDegreePrior(m_labelGraphPrior);
            checkSafety();
            sample();
    }
    virtual ~DegreeCorrectedStochasticBlockModelFamily(){
        delete m_edgeCountPriorPtr;
        delete m_blockPriorPtr;
        delete m_degreePriorPtr;
    }
};

}// end FastMIDyNet
#endif
