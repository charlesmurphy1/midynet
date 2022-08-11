#ifndef FAST_MIDYNET_SBM_H
#define FAST_MIDYNET_SBM_H

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "prior/label_graph.h"
#include "prior/block.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/random_graph/util.h"
#include "FastMIDyNet/random_graph/likelihood/sbm.h"
#include "FastMIDyNet/proposer/edge/edge_proposer.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class StochasticBlockModelBase: public BlockLabeledRandomGraph{
protected:
    std::unique_ptr<StochasticBlockModelLikelihood> m_sbmLikelihoodModelUPtr = nullptr;
    LabelGraphPrior* m_labelGraphPriorPtr = nullptr;
    bool m_stubLabeled;

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
        m_sbmLikelihoodModelUPtr->m_statePtr = &m_state;
        m_sbmLikelihoodModelUPtr->m_withSelfLoopsPtr = &m_withSelfLoops;
        m_sbmLikelihoodModelUPtr->m_withParallelEdgesPtr = &m_withParallelEdges;
        m_sbmLikelihoodModelUPtr->m_labelGraphPriorPtrPtr = &m_labelGraphPriorPtr;
    }

    using BlockLabeledRandomGraph::BlockLabeledRandomGraph;

    StochasticBlockModelBase(size_t graphSize, bool stubLabeled=true, bool withSelfLoops=true, bool withParallelEdges=true):
        VertexLabeledRandomGraph<BlockIndex>(graphSize, withSelfLoops, withParallelEdges),
        m_stubLabeled(stubLabeled){
            m_sbmLikelihoodModelUPtr = std::unique_ptr<StochasticBlockModelLikelihood>(makeSBMLikelihood(stubLabeled));
            m_likelihoodModelPtr = m_vertexLabeledlikelihoodModelPtr = m_sbmLikelihoodModelUPtr.get();
        }

    StochasticBlockModelBase(size_t graphSize, LabelGraphPrior& prior, bool stubLabeled=true, bool withSelfLoops=true, bool withParallelEdges=true):
        VertexLabeledRandomGraph<BlockIndex>(graphSize, withSelfLoops, withParallelEdges),
        m_stubLabeled(stubLabeled){
            m_sbmLikelihoodModelUPtr = std::unique_ptr<StochasticBlockModelLikelihood>(makeSBMLikelihood(stubLabeled));
            m_likelihoodModelPtr = m_vertexLabeledlikelihoodModelPtr = m_sbmLikelihoodModelUPtr.get();
            setLabelGraphPrior(prior);
        }
public:

    void sampleLabels() override {
        m_labelGraphPriorPtr->samplePartition();
    }

    void setState(const MultiGraph& state) override{
        RandomGraph::setState(state);
        m_labelGraphPriorPtr->setGraph(m_state);
    }
    void setLabels(const std::vector<BlockIndex>& labels) override { m_labelGraphPriorPtr->setPartition(labels); }


    LabelGraphPrior& getLabelGraphPriorRef() const { return *m_labelGraphPriorPtr; }
    const LabelGraphPrior& getLabelGraphPrior() const { return *m_labelGraphPriorPtr; }
    void setLabelGraphPrior(LabelGraphPrior& labelGraphPrior) {
        m_labelGraphPriorPtr = &labelGraphPrior;
        m_labelGraphPriorPtr->isRoot(false);
        setUpLikelihood();

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

    void checkSelfConsistency() const override{
        VertexLabeledRandomGraph<BlockIndex>::checkSelfConsistency();
        m_labelGraphPriorPtr->checkSelfConsistency();
        checkGraphConsistencyWithLabelGraph("StochasticBlockModelBase", m_state, getLabels(), getLabelGraph());
    }
    const bool isCompatible(const MultiGraph& graph) const override{
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
    LabelGraphDeltaPrior m_labelGraphPrior;
    std::unique_ptr<EdgeProposer> m_edgeProposerUPtr = nullptr;
    std::unique_ptr<LabelProposer<BlockIndex>> m_labelProposerUPtr;
public:
    StochasticBlockModel(
        const BlockSequence& blocks,
        const LabelGraph& labelGraph,
        bool stubLabeled=true,
        bool withSelfLoops=true,
        bool withParallelEdges=true,
        std::string edgeProposerType="uniform"):
        StochasticBlockModelBase(blocks.size(), stubLabeled, withSelfLoops, withParallelEdges),
        m_blockPrior(blocks),
        m_labelGraphPrior(labelGraph){
                m_edgeProposerUPtr = std::unique_ptr<EdgeProposer>(makeEdgeProposer(edgeProposerType, false, false, withSelfLoops, withSelfLoops));
                m_edgeProposerPtr = m_edgeProposerUPtr.get();
                m_edgeProposerPtr->isRoot(false);

                m_labelProposerUPtr = std::unique_ptr<LabelProposer<BlockIndex>>( makeBlockProposer("uniform") );
                m_labelProposerPtr = m_labelProposerUPtr.get();
                m_labelProposerPtr->isRoot(false);

                checkSafety();
                sample();
            }
};

class StochasticBlockModelFamily: public StochasticBlockModelBase{
    std::unique_ptr<BlockCountPrior> m_blockCountPriorUPtr = nullptr;
    std::unique_ptr<BlockPrior> m_blockPriorUPtr = nullptr;
    std::unique_ptr<EdgeCountPrior> m_edgeCountPriorUPtr = nullptr;
    std::unique_ptr<LabelGraphPrior> m_labelGraphPriorUPtr = nullptr;
    std::unique_ptr<EdgeProposer> m_edgeProposerUPtr = nullptr;
    std::unique_ptr<LabelProposer<BlockIndex>> m_labelProposerUPtr = nullptr;
public:
    StochasticBlockModelFamily(
        size_t size,
        double edgeCount,
        size_t blockCount=0,
        bool useHyperPrior=true,
        bool usePlantedPrior=true,
        bool canonical=false,
        bool stubLabeled=true,
        bool withSelfLoops=true,
        bool withParallelEdges=true,
        std::string edgeProposerType="uniform",
        std::string blockProposerType="uniform",
        double sampleLabelCountProb=0.1,
        double labelCreationProb=0.5,
        double shift=1
    ):
        StochasticBlockModelBase(size, stubLabeled, withSelfLoops, withParallelEdges){
            if (blockCount < 1 or blockCount > size-1)
                m_blockCountPriorUPtr = std::unique_ptr<BlockCountPrior>(new BlockCountUniformPrior(1, size-1));
            else {
                m_blockCountPriorUPtr = std::unique_ptr<BlockCountPrior>(new BlockCountDeltaPrior(blockCount));
                sampleLabelCountProb = 0;
            }
            if (stubLabeled)
                withSelfLoops = withParallelEdges = true;

            m_edgeCountPriorUPtr = std::unique_ptr<EdgeCountPrior>(makeEdgeCountPrior(edgeCount, canonical));
            m_blockPriorUPtr = std::unique_ptr<BlockPrior>(makeBlockPrior(size, *m_blockCountPriorUPtr, useHyperPrior));
            m_labelGraphPriorUPtr = std::unique_ptr<LabelGraphPrior>( makeLabelGraphPrior(*m_edgeCountPriorUPtr, *m_blockPriorUPtr, usePlantedPrior) );
            setLabelGraphPrior(*m_labelGraphPriorUPtr);

            m_edgeProposerUPtr = std::unique_ptr<EdgeProposer>(
                makeEdgeProposer(edgeProposerType, canonical, false, withSelfLoops, withParallelEdges)
            );
            setEdgeProposer(*m_edgeProposerUPtr);
            m_labelProposerUPtr = std::unique_ptr<LabelProposer<BlockIndex>>(
                makeBlockProposer(blockProposerType, useHyperPrior, sampleLabelCountProb, labelCreationProb, shift)
            );
            setLabelProposer(*m_labelProposerUPtr);

            checkSafety();
            sample();
    }
};

class PlantedPartitionModel: public StochasticBlockModelBase{

};

}// end FastMIDyNet
#endif
