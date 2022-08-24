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


    DegreeCorrectedStochasticBlockModelBase(size_t graphSize):
        VertexLabeledRandomGraph<BlockIndex>(graphSize, m_likelihoodModel, true, true) { }
    DegreeCorrectedStochasticBlockModelBase(size_t graphSize, VertexLabeledDegreePrior& degreePrior):
        VertexLabeledRandomGraph<BlockIndex>(graphSize, m_likelihoodModel, true, true){
            setDegreePrior(degreePrior);
            m_degreePriorPtr->isRoot(false);
        }

public:

    void sampleLabels() override {
        m_degreePriorPtr->samplePartition();
    }

    void setState(const MultiGraph& state) override {
        RandomGraph::setState(state);
        m_degreePriorPtr->setGraph(m_state);
    }
    void setLabels(const std::vector<BlockIndex>& labels, bool reduce=false) override {
        m_degreePriorPtr->setPartition(labels);
        if ( reduce )
            reduceLabels();
    }


    const VertexLabeledDegreePrior& getDegreePrior() const { return *m_degreePriorPtr; }
    VertexLabeledDegreePrior& getDegreePriorRef() const { return *m_degreePriorPtr; }
    void setDegreePrior(VertexLabeledDegreePrior& prior) {
        prior.isRoot(false);
        m_degreePriorPtr = &prior;
        setUpLikelihood();
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
    const double getLabelLogJoint() const override {
        return m_degreePriorPtr->getBlockPrior().getLogJoint();
    }
    void reduceLabels() override {
        m_degreePriorPtr->reducePartition();
        setUp();
    }

    virtual void checkSelfConsistency() const override {
        VertexLabeledRandomGraph<BlockIndex>::checkSelfConsistency();
        m_degreePriorPtr->checkSelfConsistency();
        checkGraphConsistencyWithLabelGraph(
            "DegreeCorrectedStochasticBlockModelBase", m_state, getLabels(), getLabelGraph()
        );
        checkGraphConsistencyWithDegreeSequence(
            "DegreeCorrectedStochasticBlockModelBase", "m_state", m_state, "m_degreePriorPtr", getDegrees()
        );
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
    std::unique_ptr<BlockCountPrior> m_blockCountPriorUPtr;
    std::unique_ptr<LabelGraphPrior> m_labelGraphPriorUPtr;

    std::unique_ptr<BlockPrior> m_blockPriorUPtr;
    std::unique_ptr<EdgeCountPrior> m_edgeCountPriorUPtr;
    std::unique_ptr<VertexLabeledDegreePrior> m_degreePriorUPtr;
    std::unique_ptr<EdgeProposer> m_edgeProposerUPtr;
    std::unique_ptr<LabelProposer<BlockIndex>> m_labelProposerUPtr;
public:
    DegreeCorrectedStochasticBlockModelFamily(
        size_t size,
        double edgeCount,
        size_t blockCount=0,
        bool useBlockHyperPrior=false,
        bool useDegreeHyperPrior=false,
        bool usePlantedPrior=false,
        bool canonical=false,
        std::string edgeProposerType="degree",
        std::string blockProposerType="uniform",
        double sampleLabelCountProb=0.1,
        double labelCreationProb=0.5,
        double shift=1
    ):
        DegreeCorrectedStochasticBlockModelBase(size){
            if (blockCount == 0)
                m_blockCountPriorUPtr = std::unique_ptr<BlockCountPrior>(new BlockCountUniformPrior(1, size-1));
            else{
                m_blockCountPriorUPtr = std::unique_ptr<BlockCountPrior>(new BlockCountDeltaPrior(blockCount));
                sampleLabelCountProb = 0;
            }
            m_edgeCountPriorUPtr = std::unique_ptr<EdgeCountPrior>(makeEdgeCountPrior(edgeCount, canonical));
            m_blockPriorUPtr = std::unique_ptr<BlockPrior>(makeBlockPrior(size, *m_blockCountPriorUPtr, useBlockHyperPrior));
            m_labelGraphPriorUPtr = std::unique_ptr<LabelGraphPrior>( makeLabelGraphPrior(*m_edgeCountPriorUPtr, *m_blockPriorUPtr, usePlantedPrior) );
            m_degreePriorUPtr = std::unique_ptr<VertexLabeledDegreePrior>(makeVertexLabeledDegreePrior(*m_labelGraphPriorUPtr, useDegreeHyperPrior));
            setDegreePrior(*m_degreePriorUPtr);

            m_edgeProposerUPtr = std::unique_ptr<EdgeProposer>(
                makeEdgeProposer(edgeProposerType, canonical, false, true, true)
            );
            setEdgeProposer(*m_edgeProposerUPtr);

            m_labelProposerUPtr = std::unique_ptr<LabelProposer<BlockIndex>>(
                makeBlockProposer(blockProposerType, useBlockHyperPrior, sampleLabelCountProb, labelCreationProb, shift)
            );
            setLabelProposer(*m_labelProposerUPtr);

            checkSafety();
            sample();
    }
};

}// end FastMIDyNet
#endif
