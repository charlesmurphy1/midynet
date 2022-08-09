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

    ConfigurationModelBase(size_t graphSize):
        RandomGraph(graphSize, m_likelihoodModel){ setUpLikelihood(); }
    ConfigurationModelBase(size_t graphSize, DegreePrior& degreePrior):
        RandomGraph(graphSize, m_likelihoodModel), m_degreePriorPtr(&degreePrior){
            setUpLikelihood();
            m_degreePriorPtr->isRoot(false);
            m_degreePriorPtr->setSize(m_size);
        }
public:

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
    std::unique_ptr<DegreePrior> m_degreePriorUPtr = nullptr;
    std::unique_ptr<EdgeProposer> m_edgeProposerUPtr = nullptr;
public:
    ConfigurationModel(const DegreeSequence& degrees):
        ConfigurationModelBase(degrees.size()) {
            m_degreePriorUPtr = std::unique_ptr<DegreePrior>(new DegreeDeltaPrior(degrees));
            m_degreePriorPtr = m_degreePriorUPtr.get();
            m_degreePriorPtr->isRoot(false);

            m_edgeProposerUPtr = std::unique_ptr<EdgeProposer>(makeEdgeProposer("degree", false, true, true, true));
            m_edgeProposerPtr = m_edgeProposerUPtr.get();
            m_edgeProposerPtr->isRoot(false);

            checkSafety();
            sample();
        }
};

class ConfigurationModelFamily : public ConfigurationModelBase{
    std::unique_ptr<EdgeCountPrior> m_edgeCountPriorUPtr = nullptr;
    std::unique_ptr<DegreePrior> m_degreePriorUPtr = nullptr;
    std::unique_ptr<EdgeProposer> m_edgeProposerUPtr = nullptr;
public:
    ConfigurationModelFamily(
        size_t size, double edgeCount, bool useHyperPrior=true, bool canonical=false, std::string edgeProposerType="degree"
    ):
        ConfigurationModelBase(size) {
            m_edgeCountPriorUPtr = std::unique_ptr<EdgeCountPrior>(makeEdgeCountPrior(edgeCount, canonical));
            m_degreePriorUPtr = std::unique_ptr<DegreePrior>(makeDegreePrior(size, *m_edgeCountPriorUPtr, useHyperPrior));
            setDegreePrior(*m_degreePriorUPtr);

            m_edgeProposerUPtr = std::unique_ptr<EdgeProposer>(makeEdgeProposer(edgeProposerType, canonical, true, true, true));
            setEdgeProposer(*m_edgeProposerUPtr);

            checkSafety();
            sample();
        }
};

}// end FastMIDyNet
#endif
