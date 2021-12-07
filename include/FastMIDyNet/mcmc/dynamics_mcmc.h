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
    EdgeProposer& m_edgeProposer;
    RandomGraphMCMC& m_randomGraphMCMC;
    double m_betaLikelihood, m_betaPrior, m_sampleGraphPrior;
    std::uniform_real_distribution<double> m_uniform;
    bool m_lastMoveWasGraphMove;
public:
    DynamicsMCMC(
        Dynamics& dynamics,
        RandomGraphMCMC& randomGraphMCMC,
        EdgeProposer& edgeProposer,
        double betaLikelihood=1,
        double betaPrior=1,
        double sampleGraphPrior=0.5,
        const CallBackList& callBacks={}):
    MCMC(callBacks),
    m_dynamics(dynamics),
    m_randomGraphMCMC(randomGraphMCMC),
    m_edgeProposer(edgeProposer),
    m_betaLikelihood(betaLikelihood),
    m_betaPrior(betaPrior),
    m_sampleGraphPrior(sampleGraphPrior),
    m_uniform(0., 1.) {}

    void setUp();

    const MultiGraph& getGraph() { return m_dynamics.getGraph(); }

    virtual double getLogLikelihood() { return m_dynamics.getLogLikelihood(); }
    virtual double getLogPrior() { return m_dynamics.getLogPrior(); }
    virtual double getLogJoint() { return m_dynamics.getLogJoint(); }

    void doMetropolisHastingsStep();
};

}

#endif
