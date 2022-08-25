#ifndef FAST_MIDYNET_HSBM_H
#define FAST_MIDYNET_HSBM_H

#include "FastMIDyNet/random_graph/likelihood/sbm.h"
#include "FastMIDyNet/random_graph/prior/nested_label_graph.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/random_graph/util.h"

namespace FastMIDyNet{

class NestedStochasticBlockModelBase: public NestedBlockLabeledRandomGraph{
protected:
    std::unique_ptr<StochasticBlockModelLikelihood> m_sbmLikelihoodModelUPtr = nullptr;
    NestedStochasticBlockLabelGraphPrior m_nestedLabelGraphPrior;
    LabelGraphPrior* m_labelGraphPriorPtr = &m_nestedLabelGraphPrior;
    bool m_stubLabeled;

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
    void sampleOnlyPrior() override { m_nestedLabelGraphPrior.sample(); }
    void setUpLikelihood() override {
        m_sbmLikelihoodModelUPtr->m_statePtr = &m_state;
        m_sbmLikelihoodModelUPtr->m_withSelfLoopsPtr = &m_withSelfLoops;
        m_sbmLikelihoodModelUPtr->m_withParallelEdgesPtr = &m_withParallelEdges;
        m_sbmLikelihoodModelUPtr->m_labelGraphPriorPtrPtr = &m_labelGraphPriorPtr;
    }
    NestedStochasticBlockModelBase(size_t graphSize, bool stubLabeled=true, bool withSelfLoops=true, bool withParallelEdges=true):
        NestedBlockLabeledRandomGraph(graphSize, withSelfLoops, withParallelEdges),
        m_nestedLabelGraphPrior(graphSize),
        m_stubLabeled(stubLabeled){
            m_sbmLikelihoodModelUPtr = std::unique_ptr<StochasticBlockModelLikelihood>(makeSBMLikelihood(stubLabeled));
            m_likelihoodModelPtr = m_vertexLabeledlikelihoodModelPtr = m_sbmLikelihoodModelUPtr.get();
            setUpLikelihood();

        }
    NestedStochasticBlockModelBase(size_t graphSize, EdgeCountPrior& edgeCountPrior, bool stubLabeled=true, bool withSelfLoops=true, bool withParallelEdges=true):
        NestedBlockLabeledRandomGraph(graphSize, withSelfLoops, withParallelEdges),
        m_nestedLabelGraphPrior(graphSize, edgeCountPrior),
        m_stubLabeled(stubLabeled){
            m_sbmLikelihoodModelUPtr = std::unique_ptr<StochasticBlockModelLikelihood>(makeSBMLikelihood(stubLabeled));
            m_likelihoodModelPtr = m_vertexLabeledlikelihoodModelPtr = m_sbmLikelihoodModelUPtr.get();
            setUpLikelihood();
        }
        void computeConsistentState() override {
            m_nestedLabelGraphPrior.setGraph(m_state);
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
        m_nestedLabelGraphPrior.samplePartition();
    }
    void setNestedLabels(const std::vector<BlockSequence>& labels, bool reduce=false) override {
        m_nestedLabelGraphPrior.setNestedPartition(labels);
        if ( reduce )
            reduceLabels();
    }


    const BlockPrior& getBlockPrior() const { return m_labelGraphPriorPtr->getBlockPrior(); }
    const NestedBlockPrior& getNestedBlockPrior() const { return m_nestedLabelGraphPrior.getNestedBlockPrior(); }

    const LabelGraphPrior& getLabelGraphPrior() const { return *m_labelGraphPriorPtr; }
    const NestedLabelGraphPrior& getNestedLabelGraphPrior() const { return m_nestedLabelGraphPrior; }
    const double getLabelLogJoint() const override {
        return m_nestedLabelGraphPrior.getNestedBlockPrior().getLogJoint();
    }
    void reduceLabels() override {
        m_nestedLabelGraphPrior.reducePartition();
        setUp();
    }

    void setEdgeCountPrior(EdgeCountPrior& prior) { m_nestedLabelGraphPrior.setEdgeCountPrior(prior); }

    void checkSelfConsistency() const override {
        NestedVertexLabeledRandomGraph<BlockIndex>::checkSelfConsistency();
        m_nestedLabelGraphPrior.checkSelfConsistency();
        checkGraphConsistencyWithLabelGraph("NestedStochasticBlockModelBase", m_state, getLabels(), getLabelGraph());
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
    std::unique_ptr<EdgeCountPrior> m_edgeCountPriorUPtr = nullptr;
    std::unique_ptr<EdgeProposer> m_edgeProposerUPtr = nullptr;
    std::unique_ptr<NestedLabelProposer<BlockIndex>> m_nestedLabelProposerUPtr = nullptr;
public:
    NestedStochasticBlockModelFamily(
        size_t size,
        double edgeCount,
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
        NestedStochasticBlockModelBase(size, stubLabeled, withSelfLoops, withParallelEdges){
            m_edgeCountPriorUPtr = std::unique_ptr<EdgeCountPrior>(makeEdgeCountPrior(edgeCount, canonical));
            setEdgeCountPrior(*m_edgeCountPriorUPtr);

            m_edgeProposerUPtr = std::unique_ptr<EdgeProposer>(
                makeEdgeProposer(edgeProposerType, canonical, false, withSelfLoops, withParallelEdges)
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
