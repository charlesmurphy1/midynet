#ifndef FAST_MIDYNET_DYNAMICS_MCMC_H
#define FAST_MIDYNET_DYNAMICS_MCMC_H

#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"
#include "FastMIDyNet/proposer/edge_proposer/edge_proposer.h"

namespace FastMIDyNet{

class DynamicsMCMC: public MCMC{
private:
    Dynamics& m_dynamics;
    RandomGraphMCMC& m_randomGraphMCMC;
    double m_betaLikelihood, m_betaPrior, m_sampleGraphPriorProb;
    std::uniform_real_distribution<double> m_uniform;
    bool m_lastMoveWasGraphMove;
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
    void setGraph(const MultiGraph& graph) { m_dynamics.setGraph(graph); m_randomGraphMCMC.setUpProposers(); }
    const BlockSequence& getBlocks() const override { return m_randomGraphMCMC.getBlocks(); }
    const int getSize() const { return m_dynamics.getSize(); }

    double getBetaPrior() const { return m_betaPrior; }
    void setBetaPrior(double betaPrior) { m_betaPrior = betaPrior; }

    double getBetaLikelihood() const { return m_betaLikelihood; }
    void setBetaLikelihood(double betaLikelihood) { m_betaLikelihood = betaLikelihood; }

    double getSampleGraphPriorProb() const { return m_sampleGraphPriorProb; }
    void setSampleGraphPriorProb(double sampleGraphPriorProb) { m_sampleGraphPriorProb = sampleGraphPriorProb; }

    const RandomGraphMCMC& getRandomGraphMCMC() const { return m_randomGraphMCMC; }
    const Dynamics& getDynamics() const { return m_dynamics; }

    double getLogLikelihood() const override { return m_dynamics.getLogLikelihood(); }
    double getLogPrior() const override { return m_dynamics.getLogPrior(); }
    double getLogJoint() const override { return m_dynamics.getLogJoint(); }
    void sample() override {
        m_dynamics.sample();
        m_hasState=true;
    }
    void sampleState() {
        m_dynamics.sampleState();
        m_hasState=true;
    }
    void sampleGraph() {
        m_dynamics.sampleGraph();
    }
    void sampleGraphOnly() {
        m_randomGraphMCMC.sampleGraphOnly();
    }

    void doMetropolisHastingsStep() override ;
};

}

#endif
