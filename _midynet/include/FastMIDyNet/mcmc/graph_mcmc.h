#ifndef FAST_MIDYNET_GRAPH_MCMC_H
#define FAST_MIDYNET_GRAPH_MCMC_H

#include <random>

#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/proposer/edge_proposer/edge_proposer.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/mcmc/mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"
#include "FastMIDyNet/rng.h"

namespace FastMIDyNet{

class RandomGraphMCMC: public MCMC{
protected:
    RandomGraph* m_randomGraphPtr;
    EdgeProposer* m_edgeProposerPtr;
    double m_betaLikelihood, m_betaPrior;
    std::uniform_real_distribution<double> m_uniform;
public:
    RandomGraphMCMC(
        RandomGraph& randomGraph,
        EdgeProposer& edgeProposer,
        double betaLikelihood=1,
        double betaPrior=1,
        const CallBackList& callBacks={}):
    MCMC(callBacks),
    m_betaLikelihood(betaLikelihood),
    m_betaPrior(betaPrior),
    m_uniform(0., 1.) { setRandomGraph(randomGraph); setEdgeProposer(edgeProposer); }

    RandomGraphMCMC(
        double betaLikelihood=1,
        double betaPrior=1,
        const CallBackList& callBacks={}):
    MCMC(callBacks),
    m_uniform(0., 1.) {}


    const MultiGraph& getGraph() const { return m_randomGraphPtr->getState(); }
    double getBetaPrior() const { return m_betaPrior; }
    void setBetaPrior(double betaPrior) { m_betaPrior = betaPrior; }

    double getBetaLikelihood() const { return m_betaLikelihood; }
    void setBetaLikelihood(double betaLikelihood) { m_betaLikelihood = betaLikelihood; }

    const RandomGraph& getRandomGraph() const { return *m_randomGraphPtr; }
    void setRandomGraph(RandomGraph& randomGraph) { m_randomGraphPtr = &randomGraph; }

    double getLogLikelihood() const { return m_randomGraphPtr->getLogLikelihood(); }
    double getLogPrior() { return m_randomGraphPtr->getLogPrior(); }
    double getLogJoint() { return m_randomGraphPtr->getLogJoint(); }
    void sample() { m_randomGraphPtr->sample(); m_hasState=true; }

    virtual void setUp() {
        m_edgeProposerPtr->setUp(*m_randomGraphPtr);
        MCMC::setUp();
    }


    const EdgeProposer& getEdgeProposer() const { return *m_edgeProposerPtr; }
    void setEdgeProposer(EdgeProposer& edgeProposer) { m_edgeProposerPtr = &edgeProposer; }
    GraphMove proposeEdgeMove() const { return m_edgeProposerPtr->proposeMove(); }
    double getLogProposalProbRatio(const GraphMove& move) const { return m_edgeProposerPtr->getLogProposalProbRatio(move); }
    void updateProbabilities(const GraphMove& move) { m_edgeProposerPtr->updateProbabilities(move); }


};

class StochasticBlockGraphMCMC: public RandomGraphMCMC{
private:
    BlockProposer* m_blockProposerPtr;
    StochasticBlockModelFamily* m_sbmGraphPtr;
public:

    StochasticBlockGraphMCMC(
        StochasticBlockModelFamily& sbmGraph,
        EdgeProposer& edgeProposer,
        BlockProposer& blockProposer,
        double betaLikelihood=1,
        double betaPrior=1,
        const CallBackList& callBacks={}):
    RandomGraphMCMC(sbmGraph, edgeProposer, betaLikelihood, betaPrior, callBacks),
    m_blockProposerPtr(&blockProposer){}

    StochasticBlockGraphMCMC(
        double betaLikelihood=1,
        double betaPrior=1,
        const CallBackList& callBacks={}):
    RandomGraphMCMC(betaLikelihood, betaPrior, callBacks){}

    void setUp(){
        m_blockProposerPtr->setUp(*m_sbmGraphPtr);
        RandomGraphMCMC::setUp();
    };

    const BlockSequence& getBlocks() { return m_sbmGraphPtr->getBlocks(); }
    const StochasticBlockModelFamily& getRandomGraph() const { return *m_sbmGraphPtr; }
    void setRandomGraph(StochasticBlockModelFamily& randomGraph) {
        RandomGraphMCMC::setRandomGraph(randomGraph); m_sbmGraphPtr = &randomGraph;
    }

    const BlockProposer& getBlockProposer() const { return *m_blockProposerPtr; }
    void setBlockProposer(BlockProposer& blockProposer) { m_blockProposerPtr = &blockProposer; }
    BlockMove proposeBlockMove() const { return m_blockProposerPtr->proposeMove(); }


    void doMetropolisHastingsStep();

};

}

#endif
