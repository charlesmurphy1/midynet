#ifndef FAST_MIDYNET_HDCSBM_H
#define FAST_MIDYNET_HDCSBM_H

#include "FastMIDyNet/random_graph/likelihood/dcsbm.h"
#include "FastMIDyNet/random_graph/prior/nested_label_graph.h"
#include "FastMIDyNet/random_graph/prior/labeled_degree.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/random_graph/util.h"

namespace FastMIDyNet{

class NestedDegreeCorrectedStochasticBlockModelBase: public NestedBlockLabeledRandomGraph{
protected:
    DegreeCorrectedStochasticBlockModelLikelihood m_likelihoodModel;
    NestedStochasticBlockLabelGraphPrior m_nestedLabelGraphPrior;
    VertexLabeledDegreePrior* m_degreePriorPtr = nullptr;
protected:

    void _applyGraphMove (const GraphMove& move) override {
        m_degreePriorPtr->applyGraphMove(move);
        RandomGraph::_applyGraphMove(move);
    }
    void _applyLabelMove (const BlockMove& move) override {
        m_degreePriorPtr->applyLabelMove(move);
    }
    const double _getLogPrior() const override { return m_degreePriorPtr->getLogJoint(); }
    const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const override { return m_degreePriorPtr->getLogJointRatioFromGraphMove(move); }
    const double _getLogPriorRatioFromLabelMove(const BlockMove& move) const override {
        return m_degreePriorPtr->getLogJointRatioFromLabelMove(move);
    }
    void _samplePrior() override { m_degreePriorPtr->sample(); }
    void setUpLikelihood() override {
        m_likelihoodModel.m_statePtr = &m_state;
        m_likelihoodModel.m_degreePriorPtrPtr = &m_degreePriorPtr;
    }
public:

    NestedDegreeCorrectedStochasticBlockModelBase(size_t graphSize):
        NestedBlockLabeledRandomGraph(graphSize, m_likelihoodModel),
        m_nestedLabelGraphPrior(graphSize),
        m_likelihoodModel() { setUpLikelihood(); }
    NestedDegreeCorrectedStochasticBlockModelBase(size_t graphSize, EdgeCountPrior& EdgeCountPrior, VertexLabeledDegreePrior& degreePrior):
        NestedBlockLabeledRandomGraph(graphSize, m_likelihoodModel),
        m_likelihoodModel(),
        m_nestedLabelGraphPrior(graphSize, EdgeCountPrior){
            setDegreePrior(degreePrior);
            setUpLikelihood();
        }


    const size_t getEdgeCount() const { return m_nestedLabelGraphPrior.getEdgeCount(); }
    const size_t getDepth() const override { return m_nestedLabelGraphPrior.getDepth(); }
    using NestedBlockLabeledRandomGraph::getLabelOfIdx;
    const BlockIndex getLabelOfIdx(BaseGraph::VertexIndex vertex, Level level) const override { return m_nestedLabelGraphPrior.getBlockOfIdx(vertex, level); }
    const BlockIndex getNestedLabelOfIdx(BaseGraph::VertexIndex vertex, Level level) const override { return m_nestedLabelGraphPrior.getNestedBlockOfIdx(vertex, level); }
    const std::vector<std::vector<BlockIndex>>& getNestedLabels() const override { return m_nestedLabelGraphPrior.getNestedBlocks(); }
    const std::vector<BlockIndex>& getNestedLabels(Level level) const override { return m_nestedLabelGraphPrior.getNestedBlocks(level); }
    const std::vector<size_t>& getNestedLabelCount() const override { return m_nestedLabelGraphPrior.getNestedBlockCount(); }
    const size_t getNestedLabelCount(Level level) const override { return m_nestedLabelGraphPrior.getNestedBlockCount(level); }
    const std::vector<CounterMap<BlockIndex>>& getNestedVertexCounts() const override { return m_nestedLabelGraphPrior.getNestedVertexCounts(); }
    const CounterMap<BlockIndex>& getNestedVertexCounts(Level level) const override { return m_nestedLabelGraphPrior.getNestedVertexCounts(level); }
    const std::vector<CounterMap<BlockIndex>>& getNestedEdgeLabelCounts() const override { return m_nestedLabelGraphPrior.getNestedEdgeCounts(); }
    const CounterMap<BlockIndex>& getNestedEdgeLabelCounts(Level level) const override { return m_nestedLabelGraphPrior.getNestedEdgeCounts(level); }
    const std::vector<MultiGraph>& getNestedLabelGraph() const override { return m_nestedLabelGraphPrior.getNestedState(); }
    const MultiGraph& getNestedLabelGraph(Level level) const override { return m_nestedLabelGraphPrior.getNestedState(level); }



    void sampleLabels() override {
        m_degreePriorPtr->samplePartition();
    }

    void setState(const MultiGraph state) override{
        RandomGraph::setState(state);
        m_degreePriorPtr->setGraph(m_state);
    }
    void setNestedLabels(const std::vector<BlockSequence>& labels) override {
        m_nestedLabelGraphPrior.setNestedPartition(labels);
        m_degreePriorPtr->recomputeConsistentState();
    }


    const BlockPrior& getBlockPrior() const { return m_nestedLabelGraphPrior.getBlockPrior(); }
    const NestedBlockPrior& getNestedBlockPrior() const { return m_nestedLabelGraphPrior.getNestedBlockPrior(); }

    const LabelGraphPrior& getLabelGraphPrior() const { return m_nestedLabelGraphPrior; }
    const NestedLabelGraphPrior& getNestedLabelGraphPrior() const { return m_nestedLabelGraphPrior; }

    void setEdgeCountPrior(EdgeCountPrior& prior) { m_nestedLabelGraphPrior.setEdgeCountPrior(prior); }
    const VertexLabeledDegreePrior& getDegreeCountPrior() const { return *m_degreePriorPtr; }
    void setDegreePrior(VertexLabeledDegreePrior& prior) {
        m_degreePriorPtr = &prior;
        m_degreePriorPtr->setLabelGraphPrior(m_nestedLabelGraphPrior);
        m_degreePriorPtr->isRoot(false);
    }

    void checkSelfConsistency() const override {
        m_degreePriorPtr->checkSelfConsistency();
        checkGraphConsistencyWithLabelGraph("NestedDegreeStochasticBlockModelFamily", m_state, getLabels(), getLabelGraph());
    }
    const bool isCompatible(const MultiGraph& graph) const override{
        if (not VertexLabeledRandomGraph<BlockIndex>::isCompatible(graph)){
            // std::cout << "CODE 1" << std::endl;
            return false;
        }
        if (getLabelGraphFromGraph(graph, getLabels()) != getLabelGraph()){
            std::cout << "CODE 2" << std::endl;
            return false;
        }
        if (m_degreePriorPtr->getState() != graph.getDegrees()){
            // std::cout << "CODE 3" << std::endl;
            return false;
        }
        return true;
    }

    void computationFinished() const override {
        m_isProcessed = false;
        m_degreePriorPtr->computationFinished();
    }
    bool isValidLabelMove(const BlockMove& move) const override {
        return m_nestedLabelGraphPrior.getNestedBlockPrior().isValidBlockMove(move);
    }

};

class NestedNestedDegreeCorrectedStochasticBlockModelFamily: public NestedDegreeCorrectedStochasticBlockModelBase{
    EdgeCountPrior* m_edgeCountPriorPtr;
public:
    NestedNestedDegreeCorrectedStochasticBlockModelFamily(size_t size, double edgeCount, bool useHyperPrior=true, bool canonical=false):
        NestedDegreeCorrectedStochasticBlockModelBase(size){
            m_edgeCountPriorPtr = makeEdgeCountPrior(edgeCount, canonical);
            m_nestedLabelGraphPrior = NestedStochasticBlockLabelGraphPrior(size, *m_edgeCountPriorPtr);
            m_degreePriorPtr = makeVertexLabeledDegreePrior(m_nestedLabelGraphPrior, useHyperPrior);
            checkSafety();
            sample();
    }
    virtual ~NestedNestedDegreeCorrectedStochasticBlockModelFamily(){
        delete m_edgeCountPriorPtr;
        delete m_degreePriorPtr;
    }
};


}
#endif
