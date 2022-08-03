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

class StochasticBlockModelBase: public BlockLabeledRandomGraph{
protected:
    StochasticBlockModelLikelihood* m_sbmLikelihoodModelPtr = nullptr;
    LabelGraphPrior* m_labelGraphPriorPtr = nullptr;
    bool m_withSelfLoops, m_withParallelEdges, m_stubLabeled;

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
        m_sbmLikelihoodModelPtr->m_statePtr = &m_state;
        m_sbmLikelihoodModelPtr->m_withSelfLoopsPtr = &m_withSelfLoops;
        m_sbmLikelihoodModelPtr->m_withParallelEdgesPtr = &m_withParallelEdges;
        m_sbmLikelihoodModelPtr->m_labelGraphPriorPtrPtr = &m_labelGraphPriorPtr;
    }

public:
    using BlockLabeledRandomGraph::BlockLabeledRandomGraph;
    StochasticBlockModelBase(size_t graphSize, bool stubLabeled=true, bool withSelfLoops=true, bool withParallelEdges=true):
        VertexLabeledRandomGraph<BlockIndex>(graphSize),
        m_stubLabeled(stubLabeled),
        m_withSelfLoops(withSelfLoops),
        m_withParallelEdges(withParallelEdges){
                m_likelihoodModelPtr = m_vertexLabeledlikelihoodModelPtr = m_sbmLikelihoodModelPtr = makeSBMLikelihood(stubLabeled);
                setUpLikelihood();
            }
    virtual ~StochasticBlockModelBase() { delete m_sbmLikelihoodModelPtr; }

    void sampleLabels() override {
        m_labelGraphPriorPtr->samplePartition();
    }

    void setState(const MultiGraph state) override{
        RandomGraph::setState(state);
        m_labelGraphPriorPtr->setGraph(m_state);
    }
    void setLabels(const std::vector<BlockIndex>& labels) override { m_labelGraphPriorPtr->setPartition(labels); }


    const BlockPrior& getBlockPrior() const { return m_labelGraphPriorPtr->getBlockPrior(); }
    void setBlockPrior(BlockPrior& blockPrior) {
        if (m_labelGraphPriorPtr == nullptr)
            throw SafetyError("StochasticBlockModelBase", "m_labelGraphPriorPtr");
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
    const bool isStubLabeled() const { return m_stubLabeled; }
    const bool withSelfLoops() const { return m_withSelfLoops; }
    const bool withParallelEdges() const { return m_withParallelEdges; }

    virtual void checkSelfConsistency() const override{
        m_labelGraphPriorPtr->checkSelfConsistency();
        checkGraphConsistencyWithLabelGraph("StochasticBlockModelBase", m_state, getLabels(), getLabelGraph());
    }
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
            throw SafetyError("StochasticBlockModelBase", "m_labelGraphPriorPtr");
    }
};


class StochasticBlockModel: public StochasticBlockModelBase{
    BlockDeltaPrior m_blockPrior;
    LabelGraphDeltaPrior m_labelGraph;

public:
    StochasticBlockModel(const BlockSequence& blocks, const LabelGraph& labelGraph, bool stubLabeled=true, bool withSelfLoops=true, bool withParallelEdges=true):
        StochasticBlockModelBase(blocks.size(), stubLabeled, withSelfLoops, withParallelEdges),
        m_blockPrior(blocks),
        m_labelGraph(labelGraph){
                checkSafety();
                sample();
            }
};

class StochasticBlockModelFamily: public StochasticBlockModelBase{
    BlockCountUniformPrior m_blockCountPrior;
    BlockPrior* m_blockPriorPtr;
    EdgeCountPrior* m_edgeCountPriorPtr;
public:
    StochasticBlockModelFamily(size_t size, double edgeCount, bool useHyperPrior=true, bool canonical=false, bool stubLabeled=true, bool withSelfLoops=true, bool withParallelEdges=true):
        StochasticBlockModelBase(size, stubLabeled, withSelfLoops, withParallelEdges),
        m_blockCountPrior(1, size-1){
            m_edgeCountPriorPtr = makeEdgeCountPrior(edgeCount, canonical);
            m_blockPriorPtr = makeBlockPrior(size, m_blockCountPrior);
            m_labelGraphPriorPtr = new LabelGraphErdosRenyiPrior(*m_edgeCountPriorPtr, *m_blockPriorPtr);
            checkSafety();
            sample();
    }
    virtual ~StochasticBlockModelFamily(){
        delete m_labelGraphPriorPtr;
        delete m_blockPriorPtr;
        delete m_edgeCountPriorPtr;
    }
};

}// end FastMIDyNet
#endif
