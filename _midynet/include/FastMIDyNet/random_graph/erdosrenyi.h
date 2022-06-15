#ifndef FAST_MIDYNET_ERDOSRENYI_H
#define FAST_MIDYNET_ERDOSRENYI_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/random_graph/random_graph.h"
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
    const std::vector<BlockIndex> m_blocks;
    const size_t m_blockCount = 1;
    const CounterMap<size_t> m_vertexCounts;
    Matrix<size_t> m_edgeMatrix = Matrix<size_t>(1, {1, 0});
    std::vector<size_t> m_edgeCounts = {1, 0};
    std::vector<size_t> m_degrees;
    std::vector<CounterMap<size_t>> m_degreeCounts = std::vector<CounterMap<size_t>>(1);

    EdgeCountPrior* m_edgeCountPriorPtr = nullptr;
protected:
    void _applyGraphMove(const GraphMove& move) override {
        m_edgeCountPriorPtr->applyGraphMove(move);
        RandomGraph::_applyGraphMove(move);
    }
public:
    SimpleErdosRenyiFamily(size_t graphSize):
        RandomGraph(graphSize),
        m_blocks(graphSize, 0),
        m_vertexCounts({0}, {graphSize}, 0),
        m_degrees(graphSize, 0){ }
    SimpleErdosRenyiFamily(size_t graphSize, EdgeCountPrior& edgeCountPrior):
        RandomGraph(graphSize),
        m_blocks(graphSize, 0),
        m_vertexCounts({0}, {graphSize}, 0),
        m_degrees(graphSize, 0)
        { setEdgeCountPrior(edgeCountPrior); }
    const std::vector<BlockIndex>& getBlocks() const override { return m_blocks; }
    const size_t& getBlockCount() const override { return m_blockCount; }
    const CounterMap<size_t>& getVertexCountsInBlocks() const override { return m_vertexCounts; }
    const Matrix<size_t>& getEdgeMatrix() const override { return m_edgeMatrix; }
    const std::vector<size_t>& getEdgeCountsInBlocks() const override { return m_edgeCounts; }
    const size_t& getEdgeCount() const override { return m_edgeCountPriorPtr->getState(); }
    const std::vector<size_t>& getDegrees() const override { return m_degrees; }
    const std::vector<CounterMap<size_t>>& getDegreeCountsInBlocks() const override {return m_degreeCounts; }

    void setGraph(const MultiGraph& graph) override{
        RandomGraph::setGraph(graph);
        m_degrees = graph.getDegrees();
        m_degreeCounts[0].clear();
        for (auto k : m_degrees)
            m_degreeCounts[0].increment(k);
        #if DEBUG
        checkSelfConsistency();
        #endif
    }

    void sampleGraph() override { setGraph(generateSER(m_size, getEdgeCount())); }
    void samplePriors() override { m_edgeCountPriorPtr->sample(); m_edgeMatrix[0][0] = getEdgeCount();}
    const double getLogLikelihood() const override { return -logBinomialCoefficient( m_size * (m_size - 1) / 2, getEdgeCount()); }
    const double getLogPrior() const override { return m_edgeCountPriorPtr->getLogJoint(); }
    const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const override{
        int edgeCountDiff = move.addedEdges.size() - move.removedEdges.size();
        return -logBinomialCoefficient( m_size * (m_size - 1) / 2, getEdgeCount() + edgeCountDiff)
               +logBinomialCoefficient( m_size * (m_size - 1) / 2, getEdgeCount());
    };
    const double getLogLikelihoodRatioFromBlockMove (const BlockMove& move) const override { return 0; }
    const double getLogPriorRatioFromGraphMove (const GraphMove& move) const override {
        return m_edgeCountPriorPtr->getLogJointRatioFromGraphMove(move);
    }
    const double getLogPriorRatioFromBlockMove (const BlockMove& move) const override { return 0; }
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
