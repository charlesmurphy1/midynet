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
    Dynamics& m_dynamics;
    RandomGraphMCMC& m_randomGraphMCMC;
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
    m_dynamics(dynamics),
    m_randomGraphMCMC(randomGraphMCMC),
    m_betaLikelihood(betaLikelihood),
    m_betaPrior(betaPrior),
    m_sampleGraphPriorProb(sampleGraphPriorProb),
    m_uniform(0., 1.) { m_randomGraphMCMC.setRandomGraph(m_dynamics.getRandomGraphRef()); }

    void setUp() override {
        m_randomGraphMCMC.setRandomGraph(m_dynamics.getRandomGraphRef());
        m_randomGraphMCMC.setUp();
        MCMC::setUp();
    };

    const MultiGraph& getGraph() const override { return m_randomGraphMCMC.getGraph(); }
    void setGraph(const MultiGraph& graph) { m_dynamics.setGraph(graph); setUp(); }
    const BlockSequence& getBlocks() const override { return m_randomGraphMCMC.getBlocks(); }
    const int getSize() const { return m_dynamics.getSize(); }

    const double getBetaPrior() const { return m_betaPrior; }
    void setBetaPrior(double betaPrior) { m_betaPrior = betaPrior; }

    const double getBetaLikelihood() const { return m_betaLikelihood; }
    void setBetaLikelihood(double betaLikelihood) { m_betaLikelihood = betaLikelihood; }

    const double getSampleGraphPriorProb() const { return m_sampleGraphPriorProb; }
    void setSampleGraphPriorProb(double sampleGraphPriorProb) { m_sampleGraphPriorProb = sampleGraphPriorProb; }

    const RandomGraphMCMC& getRandomGraphMCMC() const { return m_randomGraphMCMC; }
    const Dynamics& getDynamics() const { return m_dynamics; }

    const double getLogLikelihood() const override { return m_dynamics.getLogLikelihood(); }
    const double getLogPrior() const override { return m_dynamics.getLogPrior(); }
    const double getLogJoint() const override { return m_dynamics.getLogJoint(); }
    const std::map<BaseGraph::Edge, size_t>& getAddedEdgeCounter() const { return m_addedEdgeCounter; }
    const std::map<BaseGraph::Edge, size_t>& getRemovedEdgeCounter() const { return m_removedEdgeCounter; }
    bool doMetropolisHastingsStep() override ;

    void checkSafety() const override {
        m_dynamics.checkSafety();
        m_randomGraphMCMC.checkSafety();
    }
};

}

#endif
