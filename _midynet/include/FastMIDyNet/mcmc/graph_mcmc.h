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
    EdgeProposer* m_edgeProposerPtr = nullptr;
    BlockProposer* m_blockProposerPtr = nullptr;
    double m_betaLikelihood, m_betaPrior;
    mutable std::uniform_real_distribution<double> m_uniform;
public:
    RandomGraphMCMC(
        RandomGraph& randomGraph,
        EdgeProposer& edgeProposer,
        BlockProposer& blockProposer,
        double betaLikelihood=1,
        double betaPrior=1):
    m_betaLikelihood(betaLikelihood),
    m_betaPrior(betaPrior),
    m_uniform(0., 1.) {
        setRandomGraph(randomGraph);
        setEdgeProposer(edgeProposer);
        setBlockProposer(blockProposer);
    }

    RandomGraphMCMC(
        double betaLikelihood=1,
        double betaPrior=1):
    m_uniform(0., 1.) {}


    const MultiGraph& getGraph() const override { return m_randomGraphPtr->getGraph(); }
    void setGraph(const MultiGraph& graph) { m_randomGraphPtr->setGraph(graph); setUp(); }
    const BlockSequence& getBlocks() const override { return m_randomGraphPtr->getBlocks(); }
    double getBetaPrior() const { return m_betaPrior; }
    void setBetaPrior(double betaPrior) { m_betaPrior = betaPrior; }

    double getBetaLikelihood() const { return m_betaLikelihood; }
    void setBetaLikelihood(double betaLikelihood) { m_betaLikelihood = betaLikelihood; }

    const RandomGraph& getRandomGraph() const { return *m_randomGraphPtr; }
    RandomGraph& getRandomGraphRef() const { return *m_randomGraphPtr; }
    void setRandomGraph(RandomGraph& randomGraph) { m_randomGraphPtr = &randomGraph; m_randomGraphPtr->isRoot(false);}

    const double getLogLikelihood() const override { return m_randomGraphPtr->getLogLikelihood(); }
    const double getLogPrior() const override { return m_randomGraphPtr->getLogPrior(); }
    const double getLogJoint() const override { return m_randomGraphPtr->getLogJoint(); }

    void setUp() override {
        m_edgeProposerPtr->setUp(*m_randomGraphPtr);
        m_blockProposerPtr->setUp(*m_randomGraphPtr);
        MCMC::setUp();
    }

    const EdgeProposer& getEdgeProposer() const { return *m_edgeProposerPtr; }
    EdgeProposer& getEdgeProposerRef() const { return *m_edgeProposerPtr; }
    void setEdgeProposer(EdgeProposer& proposer) { m_edgeProposerPtr = &proposer; m_edgeProposerPtr->isRoot(false); }
    GraphMove proposeEdgeMove() const { return m_edgeProposerPtr->proposeMove(); }

    const BlockProposer& getBlockProposer() const { return *m_blockProposerPtr; }
    BlockProposer& getBlockProposerRef() const { return *m_blockProposerPtr; }
    void setBlockProposer(BlockProposer& proposer) { m_blockProposerPtr = &proposer; m_blockProposerPtr->isRoot(false); }


    double getLogProposalProbRatioFromGraphMove(const GraphMove& move) const { return m_edgeProposerPtr->getLogProposalProbRatio(move); }
    double getLogProposalProbRatioFromBlockMove(const BlockMove& move) const { return m_blockProposerPtr->getLogProposalProbRatio(move); }
    double _getLogAcceptanceProb(const BlockMove& move) const;
    double getLogAcceptanceProb(const BlockMove& move) const {
        return processRecursiveConstFunction<double>([&](){ return _getLogAcceptanceProb(move); }, 0) ;
    }

    void applyGraphMove(const GraphMove& move) {
        processRecursiveFunction([&](){
            m_blockProposerPtr->applyGraphMove(move);
            m_edgeProposerPtr->applyGraphMove(move);
            m_randomGraphPtr->applyGraphMove(move);
        });
    }
    void applyBlockMove(const BlockMove& move) {
        processRecursiveFunction([&](){
            m_blockProposerPtr->applyBlockMove(move);
            m_edgeProposerPtr->applyBlockMove(move);
            m_randomGraphPtr->applyBlockMove(move);
        });
    }

    bool _doMetropolisHastingsStep() override ;

    bool isSafe() const override {
        return (m_randomGraphPtr != nullptr) and (m_randomGraphPtr->isSafe())
           and (m_edgeProposerPtr != nullptr)  and (m_edgeProposerPtr->isSafe())
           and (m_blockProposerPtr != nullptr) and (m_blockProposerPtr->isSafe());
    }

    void checkSelfSafety() const override {
        if (m_randomGraphPtr == nullptr)
            throw SafetyError("RandomGraphMCMC: it is unsafe to set up, since `m_randomGraphPtr` is NULL.");
        m_randomGraphPtr->checkSafety();

        if (m_edgeProposerPtr == nullptr)
            throw SafetyError("RandomGraphMCMC: it is unsafe to set up, since `m_edgeProposerPtr` is NULL.");
        m_edgeProposerPtr->checkSafety();

        if (m_blockProposerPtr == nullptr)
            throw SafetyError("RandomGraphMCMC: it is unsafe to set up, since `m_blockProposerPtr` is NULL.");
        m_blockProposerPtr->checkSafety();
    };

    void checkSelfConsistency() const override {
        if (m_edgeProposerPtr != nullptr)
            m_edgeProposerPtr->checkConsistency();
        if (m_blockProposerPtr != nullptr)
            m_blockProposerPtr->checkConsistency();
    }

    void computationFinished() const override {
        m_isProcessed = false;
        m_blockProposerPtr->computationFinished();
        m_edgeProposerPtr->computationFinished();
        m_randomGraphPtr->computationFinished();
    }


};

}

#endif
