#ifndef FAST_MIDYNET_CONFIGURATION_H
#define FAST_MIDYNET_CONFIGURATION_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/likelihood/configuration.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/generators.h"

namespace FastMIDyNet{


class ConfigurationModelFamily: public RandomGraph{
protected:
    ConfigurationModelLikelihood m_likelihoodModel;
    DegreePrior* m_degreePriorPtr;

    void _applyGraphMove (const GraphMove& move) override {
        m_degreePriorPtr->applyGraphMove(move);
        RandomGraph::_applyGraphMove(move);
    }
    const double _getLogPrior() const override { return m_degreePriorPtr->getLogJoint(); }
    const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const override {
        return m_degreePriorPtr->getLogJointRatioFromGraphMove(move);
    }
    void _samplePrior() override { m_degreePriorPtr->sample(); }
    void setUpLikelihood() {
        m_likelihoodModel.m_graphPtr = &m_graph;
        m_likelihoodModel.m_degreePriorPtrPtr = &m_degreePriorPtr;
    }
public:
    ConfigurationModelFamily(size_t graphSize):
        RandomGraph(graphSize, m_likelihoodModel){ setUpLikelihood(); }
    ConfigurationModelFamily(size_t graphSize, EdgeCountPrior& edgeCountPrior, DegreePrior& degreePrior):
        RandomGraph(graphSize, m_likelihoodModel), m_degreePriorPtr(&degreePrior){
            setUpLikelihood();
            m_degreePriorPtr->isRoot(false);
            m_degreePriorPtr->setSize(m_size);
            setEdgeCountPrior(edgeCountPrior);
        }

    const EdgeCountPrior& getEdgeCountPrior() const { return m_degreePriorPtr->getEdgeCountPrior(); }
    void setEdgeCountPrior(EdgeCountPrior& prior) {
        if (m_degreePriorPtr == nullptr)
            throw SafetyError("ConfigurationModelFamily", "degree prior");
        m_degreePriorPtr->setEdgeCountPrior(prior);
    }

    const DegreePrior& getDegreePrior() const { return *m_degreePriorPtr; }
    void setDegreePrior(DegreePrior& prior) {
        m_degreePriorPtr = &prior;
        m_degreePriorPtr->isRoot(false);
    }

    const size_t getEdgeCount() const { return m_degreePriorPtr->getEdgeCount(); }
    void sampleState() { setGraph(generateCM(m_degreePriorPtr->getState())); }

    const bool isCompatible(const MultiGraph& graph) const override{
        return RandomGraph::isCompatible(graph) and graph.getDegrees() == m_degreePriorPtr->getState(); ;
    }
    void computationFinished() const override {
        m_isProcessed = false;
        m_degreePriorPtr->computationFinished();
    }


    void checkSelfSafety() const override {
        RandomGraph::checkSelfSafety();
        if (not m_degreePriorPtr)
            throw SafetyError("ConfigurationModelFamily", "m_degreePriorPtr");
        m_degreePriorPtr->checkSafety();
    }


};
}// end FastMIDyNet
#endif
