#ifndef FAST_MIDYNET_CONFIGURATION_H
#define FAST_MIDYNET_CONFIGURATION_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/likelihood/configuration.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/random_graph/util.h"
#include "FastMIDyNet/generators.h"

namespace FastMIDyNet{


class ConfigurationModelBase: public RandomGraph{
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
        m_likelihoodModel.m_statePtr = &m_state;
        m_likelihoodModel.m_degreePriorPtrPtr = &m_degreePriorPtr;
    }
public:
    ConfigurationModelBase(size_t graphSize):
        RandomGraph(graphSize, m_likelihoodModel){ setUpLikelihood(); }
    ConfigurationModelBase(size_t graphSize, DegreePrior& degreePrior):
        RandomGraph(graphSize, m_likelihoodModel), m_degreePriorPtr(&degreePrior){
            setUpLikelihood();
            m_degreePriorPtr->isRoot(false);
            m_degreePriorPtr->setSize(m_size);
        }

    DegreePrior& getDegreePriorRef() const { return *m_degreePriorPtr; }
    const DegreePrior& getDegreePrior() const { return *m_degreePriorPtr; }
    void setDegreePrior(DegreePrior& prior) {
        m_degreePriorPtr = &prior;
        m_degreePriorPtr->isRoot(false);
    }

    const size_t getEdgeCount() const { return m_degreePriorPtr->getEdgeCount(); }

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
            throw SafetyError("ConfigurationModelBase", "m_degreePriorPtr");
        m_degreePriorPtr->checkSafety();
    }
};

class ConfigurationModel : public ConfigurationModelBase{
public:
    ConfigurationModel(const DegreeSequence& degrees):
        ConfigurationModelBase(degrees.size()) {
            m_degreePriorPtr = new DegreeDeltaPrior(degrees);
            checkSafety();
            sample();

        }
    virtual ~ConfigurationModel(){
        delete m_degreePriorPtr;
    }
};

class ConfigurationModelFamily : public ConfigurationModelBase{
    EdgeCountPrior* m_edgeCountPriorPtr = nullptr;
public:
    ConfigurationModelFamily(size_t size, double edgeCount, bool useHyperPrior=true, bool canonical=false):
        ConfigurationModelBase(size) {
            m_edgeCountPriorPtr = makeEdgeCountPrior(edgeCount, canonical);
            m_degreePriorPtr = makeDegreePrior(size, *m_edgeCountPriorPtr, useHyperPrior);
            setDegreePrior(*m_degreePriorPtr);
            checkSafety();
            sample();
        }
    virtual ~ConfigurationModelFamily(){
        delete m_edgeCountPriorPtr;
        delete m_degreePriorPtr;
    }
};

}// end FastMIDyNet
#endif
