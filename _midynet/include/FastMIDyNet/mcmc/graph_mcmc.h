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
    RandomGraph* m_randomGraphPtr = nullptr;
    EdgeProposer& m_edgeProposer;
    BlockProposer& m_blockProposer;
    double m_betaLikelihood, m_betaPrior;
    mutable std::uniform_real_distribution<double> m_uniform;
public:
    RandomGraphMCMC(
        RandomGraph& randomGraph,
        EdgeProposer& edgeProposer,
        BlockProposer& blockProposer,
        double betaLikelihood=1,
        double betaPrior=1,
        const CallBackList& callBacks={}):
    MCMC(callBacks),
    m_edgeProposer(edgeProposer),
    m_blockProposer(blockProposer),
    m_betaLikelihood(betaLikelihood),
    m_betaPrior(betaPrior),
    m_uniform(0., 1.) { setRandomGraph(randomGraph); }

    RandomGraphMCMC(
        EdgeProposer& edgeProposer,
        BlockProposer& blockProposer,
        double betaLikelihood=1,
        double betaPrior=1,
        const CallBackList& callBacks={}):
    MCMC(callBacks),
    m_edgeProposer(edgeProposer),
    m_blockProposer(blockProposer),
    m_uniform(0., 1.) {}


    const MultiGraph& getGraph() const override { return m_randomGraphPtr->getGraph(); }
    const BlockSequence& getBlocks() const override { return m_randomGraphPtr->getBlocks(); }
    double getBetaPrior() const { return m_betaPrior; }
    void setBetaPrior(double betaPrior) { m_betaPrior = betaPrior; }

    double getBetaLikelihood() const { return m_betaLikelihood; }
    void setBetaLikelihood(double betaLikelihood) { m_betaLikelihood = betaLikelihood; }

    const RandomGraph& getRandomGraph() const { return *m_randomGraphPtr; }
    void setRandomGraph(RandomGraph& randomGraph) { m_randomGraphPtr = &randomGraph; }

    double getLogLikelihood() const override { return m_randomGraphPtr->getLogLikelihood(); }
    double getLogPrior() const override { return m_randomGraphPtr->getLogPrior(); }
    double getLogJoint() const override { return m_randomGraphPtr->getLogJoint(); }
    void sample() override { m_randomGraphPtr->sample(); m_hasState=true; }
    void sampleGraphOnly() { m_randomGraphPtr->sampleGraph(); m_hasState=true; }

    void setUp() override {
        setUpProposers();
        MCMC::setUp();
    }

    void setUpProposers() {
        if (m_randomGraphPtr == nullptr)
            throw SafetyError("RandomGraphMCMC: it is unsafe to set up, since `m_randomGraphPtr` is NULL.");
        m_edgeProposer.setUp(*m_randomGraphPtr);
        m_blockProposer.setUp(*m_randomGraphPtr);
    }


    const EdgeProposer& getEdgeProposer() const { return m_edgeProposer; }
    GraphMove proposeEdgeMove() const { return m_edgeProposer.proposeMove(); }

    const BlockProposer& getBlockProposer() const { return m_blockProposer; }


    double getLogProposalProbRatioFromGraphMove(const GraphMove& move) const { return m_edgeProposer.getLogProposalProbRatio(move); }
    double getLogProposalProbRatioFromBlockMove(const BlockMove& move) const { return m_blockProposer.getLogProposalProbRatio(move); }
    void updateProbabilitiesFromGraphMove(const GraphMove& move) { m_blockProposer.updateProbabilities(move); m_edgeProposer.updateProbabilities(move); }
    void updateProbabilitiesFromBlockMove(const BlockMove& move) { m_blockProposer.updateProbabilities(move); m_edgeProposer.updateProbabilities(move); }

    void doMetropolisHastingsStep() override ;

};

}

#endif
