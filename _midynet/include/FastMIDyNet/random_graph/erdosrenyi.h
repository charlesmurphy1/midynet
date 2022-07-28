#ifndef FAST_MIDYNET_ERDOSRENYI_H
#define FAST_MIDYNET_ERDOSRENYI_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/random_graph/likelihood/erdosrenyi.h"
#include "FastMIDyNet/random_graph/util.h"
#include "FastMIDyNet/generators.h"

namespace FastMIDyNet{

class ErdosRenyiModelBase: public RandomGraph{
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
    ErdosRenyiModelBase(size_t graphSize, bool withSelfLoops=true, bool withParallelEdges=true):
        RandomGraph(graphSize, m_likelihoodModel),
        m_withSelfLoops(withSelfLoops),
        m_withParallelEdges(withParallelEdges){ setUpLikelihood(); }
    ErdosRenyiModelBase(size_t graphSize, EdgeCountPrior& edgeCountPrior, bool withSelfLoops=true, bool withParallelEdges=true):
        RandomGraph(graphSize, m_likelihoodModel),
        m_withSelfLoops(withSelfLoops),
        m_withParallelEdges(withParallelEdges),
        m_edgeCountPriorPtr(&edgeCountPrior){ setUpLikelihood(); }

    const size_t getEdgeCount() const { return m_edgeCountPriorPtr->getState(); }
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
    void checkSelfSafety() const override {
        RandomGraph::checkSelfSafety();
        if (not m_edgeCountPriorPtr)
            throw SafetyError("ErdosRenyiFamily", "m_edgeCountPriorPtr");
        m_edgeCountPriorPtr->checkSafety();
    }
};

class ErdosRenyiModel: public ErdosRenyiModelBase{
public:
    ErdosRenyiModel(size_t size, double edgeCount, bool withSelfLoops=true, bool withParallelEdges=true, bool canonical=false):
        ErdosRenyiModelBase(size) {
            m_edgeCountPriorPtr = makeEdgeCountPrior(edgeCount, canonical);
            setEdgeCountPrior(*m_edgeCountPriorPtr);
            checkSafety();
            sample();
        }
    ~ErdosRenyiModel(){ delete m_edgeCountPriorPtr; }
};


}// end FastMIDyNet
#endif
