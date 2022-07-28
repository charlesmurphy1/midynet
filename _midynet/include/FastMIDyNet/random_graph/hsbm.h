#ifndef FAST_MIDYNET_HSBM_H
#define FAST_MIDYNET_HSBM_H

#include "FastMIDyNet/random_graph/likelihood/sbm.h"
#include "FastMIDyNet/random_graph/prior/nested_label_graph.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/random_graph/util.h"

namespace FastMIDyNet{

class NestedStochasticBlockModelBase: public NestedBlockLabeledRandomGraph{
protected:
    StochasticBlockModelLikelihood m_likelihoodModel;
    NestedStochasticBlockLabelGraphPrior m_nestedLabelGraphPrior;
    LabelGraphPrior* m_labelGraphPriorPtr = &m_nestedLabelGraphPrior;

    void _applyGraphMove (const GraphMove& move) override {
        m_nestedLabelGraphPrior.applyGraphMove(move);
        RandomGraph::_applyGraphMove(move);
    }
    void _applyLabelMove (const BlockMove& move) override {
        m_nestedLabelGraphPrior.applyLabelMove(move);
    }
    const double _getLogPrior() const override { return m_nestedLabelGraphPrior.getLogJoint(); }
    const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const override { return m_nestedLabelGraphPrior.getLogJointRatioFromGraphMove(move); }
    const double _getLogPriorRatioFromLabelMove(const BlockMove& move) const override {
        return m_nestedLabelGraphPrior.getLogJointRatioFromLabelMove(move);
    }
    void _samplePrior() override { m_nestedLabelGraphPrior.sample(); }
    void setUpLikelihood() override {
        m_likelihoodModel.m_graphPtr = &m_graph;
        m_likelihoodModel.m_labelGraphPriorPtrPtr = &m_labelGraphPriorPtr;
    }
public:

    NestedStochasticBlockModelBase(size_t graphSize):
        NestedBlockLabeledRandomGraph(graphSize, m_likelihoodModel),
        m_nestedLabelGraphPrior(graphSize),
        m_likelihoodModel() { setUpLikelihood(); }
    NestedStochasticBlockModelBase(size_t graphSize, EdgeCountPrior& edgeCountPrior):
        NestedBlockLabeledRandomGraph(graphSize, m_likelihoodModel),
        m_likelihoodModel(),
        m_nestedLabelGraphPrior(graphSize, edgeCountPrior){
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



    void sampleState() override {
        setGraph(generateStubLabeledSBM(getLabels(), getLabelGraph()));
    }

    void sampleLabels() override {
        m_nestedLabelGraphPrior.samplePartition();
    }

    void setGraph(const MultiGraph graph) override{
        RandomGraph::setGraph(graph);
        m_nestedLabelGraphPrior.setGraph(m_graph);
    }
    void setNestedLabels(const std::vector<BlockSequence>& labels) override {
        m_nestedLabelGraphPrior.setNestedPartition(labels);
    }


    const BlockPrior& getBlockPrior() const { return m_labelGraphPriorPtr->getBlockPrior(); }
    const NestedBlockPrior& getNestedBlockPrior() const { return m_nestedLabelGraphPrior.getNestedBlockPrior(); }

    const LabelGraphPrior& getLabelGraphPrior() const { return *m_labelGraphPriorPtr; }
    const NestedLabelGraphPrior& getNestedLabelGraphPrior() const { return m_nestedLabelGraphPrior; }

    void setEdgeCountPrior(EdgeCountPrior& prior) { m_nestedLabelGraphPrior.setEdgeCountPrior(prior); }

    void checkSelfConsistency() const override {
        m_nestedLabelGraphPrior.checkSelfConsistency();
        checkGraphConsistencyWithLabelGraph("NestedStochasticBlockModelBase", m_graph, getLabels(), getLabelGraph());
    }
    const bool isCompatible(const MultiGraph& graph) const override{
        if (not VertexLabeledRandomGraph<BlockIndex>::isCompatible(graph)) return false;
        auto labelGraph = getLabelGraphFromGraph(graph, getLabels());
        return labelGraph.getAdjacencyMatrix() == getLabelGraph().getAdjacencyMatrix();
    }
    void computationFinished() const override {
        m_isProcessed = false;
        m_nestedLabelGraphPrior.computationFinished();
    }
    bool isValidLabelMove(const BlockMove& move) const override {
        return m_nestedLabelGraphPrior.getNestedBlockPrior().isValidBlockMove(move);
    }

};

class NestedStochasticBlockModelFamily: public NestedStochasticBlockModelBase{
    EdgeCountPrior* m_edgeCountPriorPtr;
public:
    NestedStochasticBlockModelFamily(size_t size, double edgeCount, bool canonical=false):
        NestedStochasticBlockModelBase(size){
            m_edgeCountPriorPtr = makeEdgeCountPrior(edgeCount, canonical);
            setEdgeCountPrior(*m_edgeCountPriorPtr);
            checkSafety();
            sample();
    }
    virtual ~NestedStochasticBlockModelFamily(){
        delete m_edgeCountPriorPtr;
    }
};



}
#endif
