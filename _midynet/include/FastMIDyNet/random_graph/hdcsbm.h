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
    void sampleOnlyPrior() override { m_degreePriorPtr->sample(); }
    void setUpLikelihood() override {
        m_likelihoodModel.m_statePtr = &m_state;
        m_likelihoodModel.m_degreePriorPtrPtr = &m_degreePriorPtr;
    }
    NestedDegreeCorrectedStochasticBlockModelBase(size_t graphSize):
        NestedBlockLabeledRandomGraph(graphSize, m_likelihoodModel, true, true),
        m_nestedLabelGraphPrior(graphSize),
        m_likelihoodModel() { }
    NestedDegreeCorrectedStochasticBlockModelBase(size_t graphSize, EdgeCountPrior& EdgeCountPrior, VertexLabeledDegreePrior& degreePrior):
        NestedBlockLabeledRandomGraph(graphSize, m_likelihoodModel, true, true),
        m_likelihoodModel(),
        m_nestedLabelGraphPrior(graphSize, EdgeCountPrior){
            setDegreePrior(degreePrior);
        }
    void computeConsistentState() override {
        m_degreePriorPtr->setGraph(m_state);
    }
public:



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



    void sampleOnlyLabels() override {
        m_degreePriorPtr->samplePartition();
    }
    void setNestedLabels(const std::vector<BlockSequence>& labels, bool reduce=false) override {
        m_nestedLabelGraphPrior.setNestedPartition(labels);
        if ( reduce )
            m_degreePriorPtr->reducePartition();
        else
            m_degreePriorPtr->recomputeConsistentState();
    }


    const BlockPrior& getBlockPrior() const { return m_nestedLabelGraphPrior.getBlockPrior(); }
    const NestedBlockPrior& getNestedBlockPrior() const { return m_nestedLabelGraphPrior.getNestedBlockPrior(); }

    const LabelGraphPrior& getLabelGraphPrior() const { return m_nestedLabelGraphPrior; }
    const NestedLabelGraphPrior& getNestedLabelGraphPrior() const { return m_nestedLabelGraphPrior; }

    void setEdgeCountPrior(EdgeCountPrior& prior) { m_nestedLabelGraphPrior.setEdgeCountPrior(prior); }
    const VertexLabeledDegreePrior& getDegreePrior() const { return *m_degreePriorPtr; }
    void setDegreePrior(VertexLabeledDegreePrior& prior) {
        m_degreePriorPtr = &prior;
        m_degreePriorPtr->setLabelGraphPrior(m_nestedLabelGraphPrior);
        m_degreePriorPtr->isRoot(false);
        setUpLikelihood();
    }
    const double getLabelLogJoint() const override {
        return m_nestedLabelGraphPrior.getNestedBlockPrior().getLogJoint();
    }

    void reduceLabels() override {
        m_degreePriorPtr->reducePartition();
        setUp();
    }
    void checkSelfConsistency() const override {
        NestedVertexLabeledRandomGraph<BlockIndex>::checkSelfConsistency();
        m_degreePriorPtr->checkSelfConsistency();
        checkGraphConsistencyWithLabelGraph("NestedDegreeStochasticBlockModelFamily", m_state, getLabels(), getLabelGraph());
    }
    const bool isCompatible(const MultiGraph& graph) const override{
        if (not VertexLabeledRandomGraph<BlockIndex>::isCompatible(graph)){
            return false;
        }
        if (getLabelGraphFromGraph(graph, getLabels()) != getLabelGraph()){
            return false;
        }
        if (m_degreePriorPtr->getState() != graph.getDegrees()){
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

class NestedDegreeCorrectedStochasticBlockModelFamily: public NestedDegreeCorrectedStochasticBlockModelBase{
    std::unique_ptr<EdgeCountPrior> m_edgeCountPriorUPtr;
    std::unique_ptr<VertexLabeledDegreePrior> m_degreePriorUPtr;
    std::unique_ptr<EdgeProposer> m_edgeProposerUPtr;
    std::unique_ptr<NestedLabelProposer<BlockIndex>> m_nestedLabelProposerUPtr;
public:
    NestedDegreeCorrectedStochasticBlockModelFamily(
        size_t size,
        double edgeCount,
        bool useDegreeHyperPrior=false,
        bool canonical=false,
        std::string edgeProposerType="uniform",
        std::string blockProposerType="uniform",
        double sampleLabelCountProb=0.1,
        double labelCreationProb=0.5,
        double shift=1
    ):
        NestedDegreeCorrectedStochasticBlockModelBase(size){
            m_edgeCountPriorUPtr = std::unique_ptr<EdgeCountPrior>(makeEdgeCountPrior(edgeCount, canonical));
            m_nestedLabelGraphPrior.setEdgeCountPrior(*m_edgeCountPriorUPtr);
            m_degreePriorUPtr = std::unique_ptr<VertexLabeledDegreePrior>(makeVertexLabeledDegreePrior(m_nestedLabelGraphPrior, useDegreeHyperPrior));
            setDegreePrior(*m_degreePriorUPtr);

            m_edgeProposerUPtr = std::unique_ptr<EdgeProposer>(
                makeEdgeProposer(edgeProposerType, canonical, false, true, true)
            );
            setEdgeProposer(*m_edgeProposerUPtr);

            m_nestedLabelProposerUPtr = std::unique_ptr<NestedLabelProposer<BlockIndex>>(
                makeNestedBlockProposer(blockProposerType, true, sampleLabelCountProb, labelCreationProb, shift)
            );
            setNestedLabelProposer(*m_nestedLabelProposerUPtr);

            checkSafety();
            sample();
    }
};


}
#endif
