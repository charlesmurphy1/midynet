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
        double sampleGraphPriorProb=0.5,
        const CallBackList& callBacks={} ):
    MCMC(callBacks),
    m_dynamicsPtr(&dynamics),
    m_randomGraphMCMCPtr(&randomGraphMCMC),
    m_betaLikelihood(betaLikelihood),
    m_betaPrior(betaPrior),
    m_sampleGraphPriorProb(sampleGraphPriorProb),
    m_uniform(0., 1.) {
        m_dynamicsPtr->isRoot(false);
        m_randomGraphMCMCPtr->isRoot(false);
        m_randomGraphMCMCPtr->setRandomGraph(m_dynamicsPtr->getRandomGraphRef());
    }

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
    // void setRandomGraphMCMC(RandomGraphMCMC& mcmc) { m_randomGraphMCMCPtr = &mcmc; }

    const Dynamics& getDynamics() const { return *m_dynamicsPtr; }
    // void setDynamics(Dynamics& dynamics) { m_dynamicsPtr = &dynamics; }

    const double getLogLikelihood() const override { return m_dynamicsPtr->getLogLikelihood(); }
    const double getLogPrior() const override { return m_dynamicsPtr->getLogPrior(); }
    const double getLogJoint() const override { return m_dynamicsPtr->getLogJoint(); }
    const std::map<BaseGraph::Edge, size_t>& getAddedEdgeCounter() const { return m_addedEdgeCounter; }
    const std::map<BaseGraph::Edge, size_t>& getRemovedEdgeCounter() const { return m_removedEdgeCounter; }

    void applyGraphMove(const GraphMove& move) {
        processRecursiveFunction([&] () {
            m_dynamicsPtr->applyGraphMove(move);
            m_randomGraphMCMCPtr->applyGraphMove(move);
        });
    }
    void applyBlockMove(const BlockMove& move) {
        processRecursiveFunction([&] () {
            m_randomGraphMCMCPtr->applyBlockMove(move);
        });
    }

    bool doMetropolisHastingsStep() override ;

    void _checkSafety() const override {
        m_dynamicsPtr->checkSafety();
        m_randomGraphMCMCPtr->checkSafety();
    }
    void _checkSelfConsistency() const override {
        m_dynamicsPtr->_checkSelfConsistency();
        m_randomGraphMCMCPtr->_checkSelfConsistency();
    }
    void computationFinished() const override {
        m_isProcessed = false;
        m_dynamicsPtr->computationFinished();
        m_randomGraphMCMCPtr->computationFinished();
    }
};

}

#endif
