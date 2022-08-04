#ifndef FAST_MIDYNET_NESTED_LABEL_GRAPH_H
#define FAST_MIDYNET_NESTED_LABEL_GRAPH_H

#include "label_graph.h"
#include "nested_block.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/utility/functions.h"

namespace FastMIDyNet{



class NestedLabelGraphPrior: public LabelGraphPrior{
protected:
    std::vector<LabelGraph> m_nestedState;
    std::vector<CounterMap<BlockIndex>> m_nestedEdgeCounts;
    NestedBlockPrior* m_nestedBlockPriorPtr = nullptr;

    void applyGraphMoveToState(const GraphMove& move) override ;
    void applyLabelMoveToState(const BlockMove& move) override ;

    void destroyBlock(const BlockMove& move) override {
        recomputeStateFromGraph();
    }

    void recomputeStateFromGraph() override ;
    std::vector<CounterMap<BlockIndex>> computeNestedEdgeCountsFromNestedState(
        const std::vector<MultiGraph>& nestedState
    ){
        std::vector<CounterMap<BlockIndex>> nestedEdgeCounts;
        for (Level l=0; l<getDepth(); ++l)
            nestedEdgeCounts.push_back(computeEdgeCountsFromState(nestedState[l]));
        return nestedEdgeCounts;
    }

    void updateNestedEdgeDiffFromEdge(
        const BaseGraph::Edge& edge, std::vector<IntMap<BaseGraph::Edge>>& nestedEdgeDiff, int counter
    ) const ;

public:
    NestedLabelGraphPrior() {}
    NestedLabelGraphPrior(EdgeCountPrior& edgeCountPrior, NestedBlockPrior& blockPrior):
        LabelGraphPrior(){
            setEdgeCountPrior(edgeCountPrior);
            setNestedBlockPrior(blockPrior);
        }
    ~NestedLabelGraphPrior(){}
    NestedLabelGraphPrior(const NestedLabelGraphPrior& other){
        setEdgeCountPrior(*other.m_edgeCountPriorPtr);
        setNestedBlockPrior(*other.m_nestedBlockPriorPtr);
        setNestedState(other.m_nestedState);
    }
    const NestedLabelGraphPrior& operator=(const NestedLabelGraphPrior& other){
        setEdgeCountPrior(*other.m_edgeCountPriorPtr);
        setNestedBlockPrior(*other.m_nestedBlockPriorPtr);
        setNestedState(other.m_nestedState);
        return *this;
    }


    virtual const LabelGraph sampleState(Level) const = 0;
    void sampleState() override ;

    virtual const double getLogLikelihoodAtLevel(Level) const = 0;
    const double getLogLikelihood() const override;

    // virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override = 0;
    // virtual const double getLogLikelihoodRatioFromLabelMoveAtLevel( const BlockMove& move ) const = 0;
    // const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const override;


    const std::vector<LabelGraph>& getNestedState() const { return m_nestedState; }
    const LabelGraph& getNestedState(Level level) const {
        return (level==-1) ? *m_graphPtr : m_nestedState[level];
    }
    void setNestedState(const std::vector<LabelGraph>& nestedState) {
        m_nestedState = nestedState;
        m_nestedEdgeCounts = computeNestedEdgeCountsFromNestedState(nestedState);
        m_state = nestedState[0];
        m_edgeCounts = m_nestedEdgeCounts[0];
        m_edgeCountPriorPtr->setState(m_state.getTotalEdgeNumber());
    }
    // void setGraph(const MultiGraph& graph) override;

    const NestedBlockPrior& getNestedBlockPrior() const{ return *m_nestedBlockPriorPtr; }
    NestedBlockPrior& getNestedBlockPriorRef() const{ return *m_nestedBlockPriorPtr; }
    void setNestedBlockPrior(NestedBlockPrior& prior) {
        setBlockPrior(prior);
        m_nestedBlockPriorPtr = &prior;
        m_nestedBlockPriorPtr->isRoot(false);
    }


    const std::vector<size_t>& getNestedBlockCount() const {
        return m_nestedBlockPriorPtr->getNestedBlockCount();
    }
    const size_t getNestedBlockCount(Level level) const {
        return m_nestedBlockPriorPtr->getNestedBlockCount(level);
    }

    const std::vector<std::vector<BlockIndex>>& getNestedBlocks() const {
        return m_nestedBlockPriorPtr->getNestedState();
    }
    const std::vector<BlockIndex>& getNestedBlocks(Level level) const {
        return m_nestedBlockPriorPtr->getNestedState(level);
    }
    using LabelGraphPrior::getBlockOfIdx;
    const BlockIndex getBlockOfIdx(BaseGraph::VertexIndex vertex, Level level) const {
        return m_nestedBlockPriorPtr->getBlockOfIdx(vertex, level);
    }
    const BlockIndex getNestedBlockOfIdx(BaseGraph::VertexIndex vertex, Level level) const {
        return m_nestedBlockPriorPtr->getNestedBlockOfIdx(vertex, level);
    }
    size_t getDepth() const { return m_nestedBlockPriorPtr->getDepth(); }


    void setNestedPartition(const std::vector<std::vector<BlockIndex>>& nestedBlocks) {
        m_nestedBlockPriorPtr->setNestedState(nestedBlocks);
        recomputeStateFromGraph();
    }
    void setPartition(const std::vector<BlockIndex>& labels) override {
        throw DepletedMethodError("NestedLabelGraphPrior", "setPartition", "setNestedPartition");
    }

    void reduceHierarchy() {
        m_nestedBlockPriorPtr->reduceHierarchy();
        recomputeStateFromGraph();
    }

    const std::vector<CounterMap<BlockIndex>>& getNestedVertexCounts() const {
        return m_nestedBlockPriorPtr->getNestedVertexCounts();
    }
    const CounterMap<BlockIndex>& getNestedVertexCounts(Level level) const {
        return m_nestedBlockPriorPtr->getNestedVertexCounts(level);
    }

    const std::vector<CounterMap<BlockIndex>>& getNestedEdgeCounts() const { return m_nestedEdgeCounts; }
    const CounterMap<BlockIndex>& getNestedEdgeCounts(Level level) const { return m_nestedEdgeCounts[level]; }

    void checkSelfConsistencyBetweenLevels() const;
    void checkSelfConsistency() const override{
        checkSelfConsistencyBetweenLevels();
        LabelGraphPrior::checkSelfConsistency();
    }
    void recomputeConsistentState() override ;
};

class NestedStochasticBlockLabelGraphPrior: public NestedLabelGraphPrior{
private:
    NestedBlockUniformHyperPrior m_blockPrior;
public:
    NestedStochasticBlockLabelGraphPrior(size_t graphSize):
        NestedLabelGraphPrior(), m_blockPrior(graphSize){
            setNestedBlockPrior(m_blockPrior);
        }
    NestedStochasticBlockLabelGraphPrior(size_t graphSize, EdgeCountPrior& edgeCountPrior):
        NestedLabelGraphPrior(), m_blockPrior(graphSize){
            setEdgeCountPrior(edgeCountPrior);
            setNestedBlockPrior(m_blockPrior);
        }
    const LabelGraph sampleState(Level level) const override ;
    const double getLogLikelihoodAtLevel(Level level) const override;
    const double getLogLikelihoodRatioFromGraphMove( const GraphMove& move ) const override ;
    const double getLogLikelihoodRatioFromLabelMove( const BlockMove& move ) const override ;
};

}

#endif
