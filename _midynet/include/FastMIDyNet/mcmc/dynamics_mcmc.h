#ifndef FAST_MIDYNET_DYNAMICS_MCMC_H
#define FAST_MIDYNET_DYNAMICS_MCMC_H

#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"
#include "FastMIDyNet/proposer/edge_proposer/edge_proposer.h"
#include "FastMIDyNet/utility/maps.hpp"

namespace FastMIDyNet{


class DynamicsMCMC: public MCMC{
private:
    Dynamics* m_dynamicsPtr = nullptr;
    RandomGraphMCMC* m_randomGraphMCMCPtr = nullptr;
    double m_betaLikelihood, m_betaPrior, m_sampleGraphPriorProb;
    std::uniform_real_distribution<double> m_uniform;
    bool m_lastMoveWasGraphMove;
    std::map<BaseGraph::Edge, size_t> m_addedEdgeCounter;
    std::map<BaseGraph::Edge, size_t> m_removedEdgeCounter;
public:
    DynamicsMCMC(
        Dynamics& dynamics,
        RandomGraphMCMC& randomGraphMCMC,
        double betaLikelihood=1,
        double betaPrior=1,
        double sampleGraphPriorProb=0.5):
    m_betaLikelihood(betaLikelihood),
    m_betaPrior(betaPrior),
    m_sampleGraphPriorProb(sampleGraphPriorProb),
    m_uniform(0., 1.) {
        setDynamics(dynamics);
        setRandomGraphMCMC(randomGraphMCMC);
    }
    DynamicsMCMC(
        double betaLikelihood=1,
        double betaPrior=1,
        double sampleGraphPriorProb=0.5):
    m_betaLikelihood(betaLikelihood),
    m_betaPrior(betaPrior),
    m_sampleGraphPriorProb(sampleGraphPriorProb),
    m_uniform(0., 1.) { }
    void setUp() override {
        m_randomGraphMCMCPtr->setRandomGraph(m_dynamicsPtr->getRandomGraphRef());
        m_randomGraphMCMCPtr->setUp();
        MCMC::setUp();
    };

    const MultiGraph& getGraph() const override { return m_randomGraphMCMCPtr->getGraph(); }
    void setGraph(const MultiGraph& graph) { m_dynamicsPtr->setGraph(graph); setUp(); }
    const BlockSequence& getBlocks() const override { return m_randomGraphMCMCPtr->getBlocks(); }
    const int getSize() const { return m_dynamicsPtr->getSize(); }

    const double getBetaPrior() const { return m_betaPrior; }
    void setBetaPrior(double betaPrior) { m_betaPrior = betaPrior; }

    const double getBetaLikelihood() const { return m_betaLikelihood; }
    void setBetaLikelihood(double betaLikelihood) { m_betaLikelihood = betaLikelihood; }

    const double getSampleGraphPriorProb() const { return m_sampleGraphPriorProb; }
    void setSampleGraphPriorProb(double sampleGraphPriorProb) { m_sampleGraphPriorProb = sampleGraphPriorProb; }

    const RandomGraphMCMC& getRandomGraphMCMC() const { return *m_randomGraphMCMCPtr; }
    RandomGraphMCMC& getRandomGraphMCMCRef() const { return *m_randomGraphMCMCPtr; }
    void setRandomGraphMCMC(RandomGraphMCMC& mcmc) {
        m_randomGraphMCMCPtr = &mcmc;
        m_randomGraphMCMCPtr->isRoot(false);
        // if (m_dynamicsPtr != nullptr and m_dynamicsPtr->isSafe())
        //     m_randomGraphMCMCPtr->setRandomGraph(m_dynamicsPtr->getRandomGraphRef());
    }

    const Dynamics& getDynamics() const { return *m_dynamicsPtr; }
    Dynamics& getDynamicsRef() const { return *m_dynamicsPtr; }
    void setDynamics(Dynamics& dynamics) {
        m_dynamicsPtr = &dynamics;
        m_dynamicsPtr->isRoot(false);
        // if (m_randomGraphMCMCPtr != nullptr and m_randomGraphMCMCPtr->isSafe())
        //     m_dynamicsPtr->setRandomGraph(m_randomGraphMCMCPtr->getRandomGraphRef());
    }

    const double getLogLikelihood() const override { return m_dynamicsPtr->getLogLikelihood(); }
    const double getLogPrior() const override { return m_dynamicsPtr->getLogPrior(); }
    const double getLogJoint() const override { return m_dynamicsPtr->getLogJoint(); }
    const std::map<BaseGraph::Edge, size_t>& getAddedEdgeCounter() const { return m_addedEdgeCounter; }
    const std::map<BaseGraph::Edge, size_t>& getRemovedEdgeCounter() const { return m_removedEdgeCounter; }
    double _getLogAcceptanceProb(const GraphMove& move) const;
    double getLogAcceptanceProb(const GraphMove& move) const{
        return processRecursiveConstFunction<double>([&](){ return _getLogAcceptanceProb(move); }, 0);
    }
    bool _doMetropolisHastingsStep() override ;

    void applyGraphMove(const GraphMove& move){
        processRecursiveFunction([&](){
            m_dynamicsPtr->applyGraphMove(move);
            m_randomGraphMCMCPtr->applyGraphMove(move);
        });
    }

    bool isSafe() const override {
        return (m_dynamicsPtr != nullptr) and (m_dynamicsPtr->isSafe())
           and (m_randomGraphMCMCPtr != nullptr) and (m_randomGraphMCMCPtr->isSafe());
    }

    void checkSelfSafety() const override {
        if (m_dynamicsPtr == nullptr)
            throw SafetyError("DynamicsMCMC: it is unsafe to set up, since `m_dynamicsPtr` is NULL.");
        m_dynamicsPtr->checkSafety();
        if (m_randomGraphMCMCPtr == nullptr)
            throw SafetyError("DynamicsMCMC: it is unsafe to set up, since `m_randomGraphMCMCPtr` is NULL.");
        m_randomGraphMCMCPtr->checkSafety();
    }
    void checkSelfConsistency() const override {
        if (m_dynamicsPtr != nullptr)
            m_dynamicsPtr->checkConsistency();
        if (m_randomGraphMCMCPtr != nullptr)
            m_randomGraphMCMCPtr->checkConsistency();
    }
    void computationFinished() const override {
        m_isProcessed = false;
        m_dynamicsPtr->computationFinished();
        m_randomGraphMCMCPtr->computationFinished();
    }
};

}

#endif
