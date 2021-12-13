#ifndef FAST_MIDYNET_GRAPH_MCMC_H
#define FAST_MIDYNET_GRAPH_MCMC_H

#include <random>

#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"
#include "FastMIDyNet/rng.h"

namespace FastMIDyNet{

class RandomGraphMCMC: public MCMC{
protected:
    RandomGraph& m_randomGraph;
    double m_betaLikelihood, m_betaPrior;
    std::uniform_real_distribution<double> m_uniform;
public:
    RandomGraphMCMC(
        RandomGraph& randomGraph,
        double betaLikelihood=1,
        double betaPrior=1,
        const CallBackList& callBacks={}):
    MCMC(callBacks),
    m_randomGraph(randomGraph),
    m_betaLikelihood(betaLikelihood),
    m_betaPrior(betaPrior),
    m_uniform(0., 1.) {}


    double getLogLikelihood() const { return m_randomGraph.getLogLikelihood(); }
    double getLogPrior() { return m_randomGraph.getLogPrior(); }
    double getLogJoint() { return m_randomGraph.getLogJoint(); }
    void sample() { m_randomGraph.sample(); m_hasState=true; }

};

class StochasticBlockGraphMCMC: public RandomGraphMCMC{
private:
    BlockProposer& m_blockProposer;
    StochasticBlockModelFamily& m_sbmGraph;
public:

    StochasticBlockGraphMCMC(
        StochasticBlockModelFamily& sbmGraph,
        BlockProposer& blockProposer,
        double betaLikelihood=1,
        double betaPrior=1,
        const CallBackList& callBacks={}):
    RandomGraphMCMC(sbmGraph, betaLikelihood, betaPrior, callBacks),
    m_sbmGraph(sbmGraph),
    m_blockProposer(blockProposer){}

    void setUp();

    const BlockSequence& getBlocks() { return m_sbmGraph.getBlocks(); }

    void doMetropolisHastingsStep();

};

}

#endif
