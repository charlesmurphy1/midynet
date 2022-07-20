#ifndef FAST_MIDYNET_ERDOSRENYI_H
#define FAST_MIDYNET_ERDOSRENYI_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/random_graph/likelihood/erdosrenyi.h"
#include "FastMIDyNet/generators.h"

namespace FastMIDyNet{

class ErdosRenyiFamily: public RandomGraph{
protected:
    EdgeCountPrior* m_edgeCountPriorPtr = nullptr;
    ErdosRenyiLikelihood m_likelihoodModel;
    bool m_withSelfLoops, m_withParallelEdges;
    void _applyGraphMove (const GraphMove& move) override {
        m_edgeCountPriorPtr->applyGraphMove(move);
        RandomGraph::_applyGraphMove(move);
    }
    const double _getLogPrior() const override { return m_edgeCountPriorPtr->getLogJoint(); }
    const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const override {
        return m_edgeCountPriorPtr->getLogJointRatioFromGraphMove(move);
    }
    void _samplePrior() override { m_edgeCountPriorPtr->sample(); }
    void setUpLikelihood() {
        m_likelihoodModel.m_graphPtr = &m_graph;
        m_likelihoodModel.m_edgeCountPriorPtrPtr = &m_edgeCountPriorPtr;
        m_likelihoodModel.m_withSelfLoopsPtr = &m_withSelfLoops;
        m_likelihoodModel.m_withParallelEdgesPtr = &m_withParallelEdges;
    }
public:
    ErdosRenyiFamily(size_t graphSize, bool withSelfLoops=true, bool withParallelEdges=true):
        RandomGraph(graphSize, m_likelihoodModel),
        m_withSelfLoops(withSelfLoops),
        m_withParallelEdges(withParallelEdges){ setUpLikelihood(); }
    ErdosRenyiFamily(size_t graphSize, EdgeCountPrior& edgeCountPrior, bool withSelfLoops=true, bool withParallelEdges=true):
        RandomGraph(graphSize, m_likelihoodModel),
        m_withSelfLoops(withSelfLoops),
        m_withParallelEdges(withParallelEdges),
        m_edgeCountPriorPtr(&edgeCountPrior){ setUpLikelihood(); }

    const size_t& getEdgeCount() const { return m_edgeCountPriorPtr->getState(); }
    void sampleState() {
        size_t E = getEdgeCount();
        const auto& generator = (m_withParallelEdges) ? generateMultiGraphErdosRenyi : generateErdosRenyi;
        setGraph(generator(m_size, E, m_withSelfLoops));
    }

    const EdgeCountPrior& getEdgeCountPrior(){ return *m_edgeCountPriorPtr; }
    void setEdgeCountPrior(EdgeCountPrior& edgeCountPrior){ m_edgeCountPriorPtr = &edgeCountPrior; }

    const bool isCompatible(const MultiGraph& graph) const override{
        return RandomGraph::isCompatible(graph) and graph.getTotalEdgeNumber() == getEdgeCount();
    }
    void computationFinished() const override {
        m_isProcessed = false;
        m_edgeCountPriorPtr->computationFinished();
    }
};

// class SimpleErdosRenyiFamily: public RandomGraph{
// private:
//     CounterMap<size_t> m_edgeCounts;
//     EdgeCountPrior* m_edgeCountPriorPtr = nullptr;
// protected:
//     void _applyGraphMove(const GraphMove& move) override {
//         m_edgeCountPriorPtr->applyGraphMove(move);
//         RandomGraph::_applyGraphMove(move);
//     }
// public:
//     SimpleErdosRenyiFamily(size_t graphSize):
//         RandomGraph(graphSize) { }
//     SimpleErdosRenyiFamily(size_t graphSize, EdgeCountPrior& edgeCountPrior):
//         RandomGraph(graphSize)
//         { setEdgeCountPrior(edgeCountPrior); }
//     const size_t& getEdgeCount() const override { return m_edgeCountPriorPtr->getState(); }
//
//     void setGraph(const MultiGraph& graph) override{
//         RandomGraph::setGraph(graph);
//         m_edgeCountPriorPtr->setState(graph.getTotalEdgeNumber());
//     }
//
//     void sample() override {
//         m_edgeCountPriorPtr->sample();
//         setGraph(generateSER(m_size, getEdgeCount()));
//         computationFinished();
//     }
//     const double getLogLikelihood() const override { return -logBinomialCoefficient( m_size * (m_size - 1) / 2, getEdgeCount()); }
//     const double getLogPrior() const override { return m_edgeCountPriorPtr->getLogJoint(); }
//     const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const override{
//         int edgeCountDiff = move.addedEdges.size() - move.removedEdges.size();
//         return -logBinomialCoefficient( m_size * (m_size - 1) / 2, getEdgeCount() + edgeCountDiff)
//                +logBinomialCoefficient( m_size * (m_size - 1) / 2, getEdgeCount());
//     };
//     const double getLogPriorRatioFromGraphMove (const GraphMove& move) const override {
//         return m_edgeCountPriorPtr->getLogJointRatioFromGraphMove(move);
//     }
//     void checkSelfConsistency() const override {
//         m_edgeCountPriorPtr->checkSelfConsistency();
//         if (m_graph.getTotalEdgeNumber() != getEdgeCount())
//             throw ConsistencyError("SimpleErdosRenyiFamily: edge count ("
//             + std::to_string(getEdgeCount()) + ") state is not equal to the number of edges in the graph ("
//             + std::to_string(m_graph.getTotalEdgeNumber()) +").");
//     }
//     void checkSelfSafety() const override {
//         if (m_edgeCountPriorPtr == nullptr)
//             throw SafetyError("SimpleErdosRenyiFamily: unsafe graph family since `m_edgeCountPriorPtr` is empty.");
//         m_edgeCountPriorPtr->checkSafety();
//     }
//
//     bool const isCompatible(const MultiGraph& graph) const override{
//         return RandomGraph::isCompatible(graph) and graph.getTotalEdgeNumber() == getEdgeCount();
//
//     }
//
//     const EdgeCountPrior& getEdgeCountPrior(){ return *m_edgeCountPriorPtr; }
//     EdgeCountPrior& getEdgeCountPriorRef(){ return *m_edgeCountPriorPtr; }
//     void setEdgeCountPrior(EdgeCountPrior& edgeCountPrior){ m_edgeCountPriorPtr = &edgeCountPrior; }
//
//     void computationFinished() const override {
//         m_isProcessed = false;
//         m_edgeCountPriorPtr->computationFinished();
//     }
//
// };

}// end FastMIDyNet
#endif
