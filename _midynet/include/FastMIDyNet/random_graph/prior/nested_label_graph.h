#ifndef FAST_MIDYNET_NESTED_LABEL_GRAPH_H
#define FAST_MIDYNET_NESTED_LABEL_GRAPH_H

#include "label_graph.h"
#include "nested_block.h"
#include "FastMIDyNet/generators.h"

namespace FastMIDyNet{



class NestedLabelGraphPrior: public LabelGraphPrior{
protected:
    std::vector<MultiGraph> m_nestedState;
    std::vector<CounterMap<BlockIndex>> m_nestedEdgeCounts;
    NestedBlockPrior* m_nestedBlockPriorPtr = nullptr;

    void applyGraphMoveToState(const GraphMove& move) override ;
    void applyLabelMoveToState(const BlockMove& move) override ;
    void recomputeConsistentState() override ;
    void recomputeStateFromGraph() override ;
    std::vector<CounterMap<BlockIndex>> computeNestedEdgeCountsFromNestedState(const std::vector<MultiGraph>& nestedState){
        std::vector<CounterMap<BlockIndex>> nestedEdgeCounts;
        for (Level l=0; l<m_nestedBlockPriorPtr->getDepth(); ++l)
            nestedEdgeCounts.push_back(computeEdgeCountsFromState(nestedState[l]));
        return nestedEdgeCounts;
    }

public:
    NestedLabelGraphPrior() {}
    NestedLabelGraphPrior(EdgeCountPrior& edgeCountPrior, NestedBlockPrior& blockPrior):
        LabelGraphPrior(edgeCountPrior, blockPrior)
        { setNestedBlockPrior(blockPrior); }
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


    virtual const MultiGraph sampleStateAtLevel(Level) const = 0;
    void sampleState() override{
        std::vector<MultiGraph> nestedLabelGraph(m_nestedBlockPriorPtr->getDepth() + 1);

        for (Level l=m_nestedBlockPriorPtr->getDepth(); l==0; --l)
            nestedLabelGraph[l] = sampleStateAtLevel(l);

        m_nestedState = nestedLabelGraph;
        m_nestedEdgeCounts = computeNestedEdgeCountsFromNestedState(m_nestedState);
        m_state = nestedLabelGraph[0];
        m_edgeCounts = m_nestedEdgeCounts[0];
    }

    const std::vector<MultiGraph>& getNestedState() const { return m_nestedState; }
    const MultiGraph& getNestedStateAtLevel(Level level) const {
        m_nestedBlockPriorPtr->checkLevel("NestedLabelGraphPrior", level);
        return (level==-1) ? *m_graphPtr : m_nestedState[level];
    }
    void setNestedState(const std::vector<MultiGraph>& nestedState) {
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
    double getLogLikelihoodRatio(size_t blockCountAfter, size_t edgeNumberAfter) const {
        return 0;
    }
    double getLogLikelihoodOfLevel( const CounterMap<BlockIndex>& vertexCounts, const MultiGraph& nextLabelGraph) const {
        return 0;
    }
public:
    using NestedLabelGraphPrior::NestedLabelGraphPrior;
    const MultiGraph sampleStateAtLevel(Level level) const override {
        if (level == m_nestedBlockPriorPtr->getDepth()){
            MultiGraph graph(1);
            graph.addMultiedgeIdx(1, 1, getEdgeCount());
            return graph;
        }
        return generateMultiGraphSBM(
                    m_nestedBlockPriorPtr->getNestedStateAtLevel(level),
                    getNestedStateAtLevel(level + 1).getAdjacencyMatrix(),
                    true
                );
    }
    const double getLogLikelihood() const override {
        return 0;
    }
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const override;
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove&) const override;


};

}

#endif
