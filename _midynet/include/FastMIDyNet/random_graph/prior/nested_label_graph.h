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
    void recomputeConsistentState() override ;
    void recomputeStateFromGraph() override ;
    std::vector<CounterMap<BlockIndex>> computeNestedEdgeCountsFromNestedState(const std::vector<MultiGraph>& nestedState){
        std::vector<CounterMap<BlockIndex>> nestedEdgeCounts;
        for (Level l=0; l<getDepth(); ++l)
            nestedEdgeCounts.push_back(computeEdgeCountsFromState(nestedState[l]));
        return nestedEdgeCounts;
    }

    void updateNestedEdgeDiffFromEdge(const BaseGraph::Edge& edge, std::vector<IntMap<BaseGraph::Edge>>& nestedEdgeDiff, int counter) const{
        size_t r = edge.first, s =edge.second;
        for (Level l=0; l<getDepth(); ++l){
            nestedEdgeDiff[l].increment({r, s}, counter);
            r = m_nestedBlockPriorPtr->getNestedStateAtLevel(l)[r];
            s = m_nestedBlockPriorPtr->getNestedStateAtLevel(l)[s];
        }
    }

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


    virtual const LabelGraph sampleStateAtLevel(Level) const = 0;
    void sampleState() override ;

    virtual const double getLogLikelihoodAtLevel(Level) const = 0;
    const double getLogLikelihood() const override{
        double logLikelihood = 0;
        for (Level l=getDepth()-1; l==0; --l)
            logLikelihood = getLogLikelihoodAtLevel(l);
        return logLikelihood;
    }

    const std::vector<LabelGraph>& getNestedState() const { return m_nestedState; }
    const LabelGraph& getNestedStateAtLevel(Level level) const {
        return (level==-1) ? *m_graphPtr : m_nestedState[level];
    }
    void setNestedState(const std::vector<LabelGraph>& nestedState) {
        m_nestedState = nestedState;
        setState(nestedState[0]);
    }

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
    const size_t getNestedBlockCountAtLevel(Level level) const {
        return m_nestedBlockPriorPtr->getNestedBlockCountAtLevel(level);
    }

    const std::vector<std::vector<BlockIndex>>& getNestedBlocks() const {
        return m_nestedBlockPriorPtr->getNestedState();
    }
    const std::vector<BlockIndex>& getNestedBlocksAtLevel(Level level) const {
        return m_nestedBlockPriorPtr->getNestedStateAtLevel(level);
    }
    const BlockIndex getBlockOfIdx(BaseGraph::VertexIndex vertex, Level level) const {
        return m_nestedBlockPriorPtr->getBlockOfIdx(vertex, level);
    }
    size_t getDepth() const { return m_nestedBlockPriorPtr->getDepth(); }


    void setNestedPartition(const std::vector<std::vector<BlockIndex>>& nestedBlocks) {
        m_nestedBlockPriorPtr->setNestedState(nestedBlocks);
        recomputeStateFromGraph();
    }

    const std::vector<CounterMap<BlockIndex>>& getNestedEdgeCounts() const { return m_nestedEdgeCounts; }

    void checkSelfConsistencyBetweenLevels() const;
    void checkSelfConsistency() const override{
        checkSelfConsistencyBetweenLevels();
        LabelGraphPrior::checkSelfConsistency();
    }
};

class NestedStochasticBlockLabelGraphPrior: public NestedLabelGraphPrior{
private:
    double getLogLikelihoodRatioOfLevel(const CounterMap<BlockIndex>& vertexCounts,
                                         const LabelGraph& nextLabelGraph,
                                         const IntMap<BaseGraph::Edge>& edgeDiff,
                                         const IntMap<BlockIndex>& vertexDiff) const ;

    double getLogLikelihoodOfLevel( const CounterMap<BlockIndex>& vertexCounts, const LabelGraph& nextLabelGraph) const ;
    NestedBlockUniformHyperPrior m_nestedBlockUniformHyperPrior;

public:
    NestedStochasticBlockLabelGraphPrior(size_t graphSize, EdgeCountPrior& edgeCountPrior):
        NestedLabelGraphPrior(), m_nestedBlockUniformHyperPrior(graphSize){
            setEdgeCountPrior(edgeCountPrior);
            setNestedBlockPrior(m_nestedBlockUniformHyperPrior);
        }
    const LabelGraph sampleStateAtLevel(Level level) const override ;
    const double getLogLikelihoodAtLevel(Level level) const override {
        return getLogLikelihoodOfLevel(m_nestedBlockPriorPtr->getNestedVertexCountsAtLevel(level), getNestedStateAtLevel(level + 1));

    }
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override ;
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const override ;


};

}

#endif
