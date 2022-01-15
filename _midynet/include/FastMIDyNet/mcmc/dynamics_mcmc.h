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
    Dynamics* m_dynamicsPtr;
    RandomGraphMCMC* m_randomGraphMCMCPtr;
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
        const CallBackList& callBacks={}):
    MCMC(callBacks),
    m_dynamicsPtr(&dynamics),
    m_randomGraphMCMCPtr(&randomGraphMCMC),
    m_betaLikelihood(betaLikelihood),
    m_betaPrior(betaPrior),
    m_sampleGraphPriorProb(sampleGraphPriorProb),
    m_uniform(0., 1.) {}

    DynamicsMCMC(
        double betaLikelihood=1,
        double betaPrior=1,
        double sampleGraphPriorProb=0.5,
        const CallBackList& callBacks={}):
    MCMC(callBacks),
    m_sampleGraphPriorProb(sampleGraphPriorProb),
    m_uniform(0., 1.) {}

    void setUp(){
        m_randomGraphMCMCPtr->setRandomGraph(m_dynamicsPtr->getRandomGraphRef());
        m_randomGraphMCMCPtr->setUp();
        MCMC::setUp();
    };

    const MultiGraph& getGraph() const { return m_dynamicsPtr->getGraph(); }
    const int getSize() const { return m_dynamicsPtr->getSize(); }

    double getBetaPrior() const { return m_betaPrior; }
    void setBetaPrior(double betaPrior) { m_betaPrior = betaPrior; }

    double getBetaLikelihood() const { return m_betaLikelihood; }
    void setBetaLikelihood(double betaLikelihood) { m_betaLikelihood = betaLikelihood; }

    double getSampleGraphPriorProb() const { return m_sampleGraphPriorProb; }
    void setSampleGraphPriorProb(double sampleGraphPriorProb) { m_sampleGraphPriorProb = sampleGraphPriorProb; }

    const RandomGraphMCMC& getRandomGraphMCMC() const { return *m_randomGraphMCMCPtr; }
    void setRandomGraphMCMC(RandomGraphMCMC& randomGraphMCMC) { m_randomGraphMCMCPtr = &randomGraphMCMC; }

    const Dynamics& getDynamics() const { return *m_dynamicsPtr; }
    void setDynamics(Dynamics& dynamics) { m_dynamicsPtr = &dynamics; }

    virtual double getLogLikelihood() const { return m_dynamicsPtr->getLogLikelihood(); }
    virtual double getLogPrior() { return m_dynamicsPtr->getLogPrior(); }
    virtual double getLogJoint() { return m_dynamicsPtr->getLogJoint(); }
    void sample() {
        m_dynamicsPtr->sample();
        m_hasState=true;
    }


    void doMetropolisHastingsStep();
};

}

#endif
