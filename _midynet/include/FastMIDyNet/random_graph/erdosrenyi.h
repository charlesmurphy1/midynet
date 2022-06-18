#ifndef FAST_MIDYNET_ERDOSRENYI_H
#define FAST_MIDYNET_ERDOSRENYI_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/generators.h"

namespace FastMIDyNet{

class ErdosRenyiFamily: public StochasticBlockModelFamily{
protected:
    const BlockSequence m_blocks;
    BlockDeltaPrior m_blockDeltaPrior;
    EdgeMatrixUniformPrior m_edgeMatrixUniformPrior;
public:
    ErdosRenyiFamily(size_t graphSize):
        StochasticBlockModelFamily(graphSize),
        m_blocks(graphSize, 0),
        m_blockDeltaPrior(m_blocks),
        m_edgeMatrixUniformPrior(){
            setBlockPrior(m_blockDeltaPrior);
            m_edgeMatrixUniformPrior.setBlockPrior(m_blockDeltaPrior);
            setEdgeMatrixPrior(m_edgeMatrixUniformPrior);
        }
    ErdosRenyiFamily(size_t graphSize, EdgeCountPrior& edgeCountPrior):
        StochasticBlockModelFamily(graphSize),
        m_blocks(graphSize, 0),
        m_blockDeltaPrior(m_blocks),
        m_edgeMatrixUniformPrior(edgeCountPrior, m_blockDeltaPrior){
            setBlockPrior(m_blockDeltaPrior);
            setEdgeMatrixPrior(m_edgeMatrixUniformPrior);

        }

    const EdgeCountPrior& getEdgeCountPrior(){ return m_edgeMatrixUniformPrior.getEdgeCountPrior(); }
    EdgeCountPrior& getEdgeCountPriorRef(){ return m_edgeMatrixUniformPrior.getEdgeCountPriorRef(); }
    void setEdgeCountPrior(EdgeCountPrior& edgeCountPrior){
        m_edgeMatrixUniformPrior.setEdgeCountPrior(edgeCountPrior);
    }

    const bool isCompatible(const MultiGraph& graph) const override{
        return RandomGraph::isCompatible(graph) and graph.getTotalEdgeNumber() == getEdgeCount();
    }
};

class SimpleErdosRenyiFamily: public RandomGraph{
private:
    CounterMap<size_t> m_edgeCounts;
    EdgeCountPrior* m_edgeCountPriorPtr = nullptr;
protected:
    void _applyGraphMove(const GraphMove& move) override {
        m_edgeCountPriorPtr->applyGraphMove(move);
        RandomGraph::_applyGraphMove(move);
    }
public:
    SimpleErdosRenyiFamily(size_t graphSize):
        RandomGraph(graphSize) { }
    SimpleErdosRenyiFamily(size_t graphSize, EdgeCountPrior& edgeCountPrior):
        RandomGraph(graphSize)
        { setEdgeCountPrior(edgeCountPrior); }
    const size_t& getEdgeCount() const override { return m_edgeCountPriorPtr->getState(); }

    void setGraph(const MultiGraph& graph) override{
        RandomGraph::setGraph(graph);
        m_edgeCountPriorPtr->setState(graph.getTotalEdgeNumber());
        #if DEBUG
        checkSelfConsistency();
        #endif
    }

    void sampleGraph() override { setGraph(generateSER(m_size, getEdgeCount())); }
    void samplePriors() override {
        m_edgeCountPriorPtr->sample();
    }
    const double getLogLikelihood() const override { return -logBinomialCoefficient( m_size * (m_size - 1) / 2, getEdgeCount()); }
    const double getLogPrior() const override { return m_edgeCountPriorPtr->getLogJoint(); }
    const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const override{
        int edgeCountDiff = move.addedEdges.size() - move.removedEdges.size();
        return -logBinomialCoefficient( m_size * (m_size - 1) / 2, getEdgeCount() + edgeCountDiff)
               +logBinomialCoefficient( m_size * (m_size - 1) / 2, getEdgeCount());
    };
    const double getLogPriorRatioFromGraphMove (const GraphMove& move) const override {
        return m_edgeCountPriorPtr->getLogJointRatioFromGraphMove(move);
    }
    void checkSelfConsistency() const override {
        m_edgeCountPriorPtr->checkSelfConsistency();
        if (m_graph.getTotalEdgeNumber() != getEdgeCount())
            throw ConsistencyError("SimpleErdosRenyiFamily: edge count ("
            + std::to_string(getEdgeCount()) + ") state is not equal to the number of edges in the graph ("
            + std::to_string(m_graph.getTotalEdgeNumber()) +").");
    }
    void checkSelfSafety() const override {
        if (m_edgeCountPriorPtr == nullptr)
            throw SafetyError("SimpleErdosRenyiFamily: unsafe graph family since `m_edgeCountPriorPtr` is empty.");
        m_edgeCountPriorPtr->checkSafety();
    }

    bool const isCompatible(const MultiGraph& graph) const override{
        return RandomGraph::isCompatible(graph) and graph.getTotalEdgeNumber() == getEdgeCount();

    }

    const EdgeCountPrior& getEdgeCountPrior(){ return *m_edgeCountPriorPtr; }
    EdgeCountPrior& getEdgeCountPriorRef(){ return *m_edgeCountPriorPtr; }
    void setEdgeCountPrior(EdgeCountPrior& edgeCountPrior){ m_edgeCountPriorPtr = &edgeCountPrior; }

    void computationFinished() const override {
        m_isProcessed = false;
        m_edgeCountPriorPtr->computationFinished();
    }

};

}// end FastMIDyNet
#endif
