#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

void StochasticBlockGraphMCMC::setUp() {
    MCMC::setUp();
    m_blockProposerPtr->setUp(*m_sbmGraphPtr);
}

void StochasticBlockGraphMCMC::doMetropolisHastingsStep() {
    BlockMove move = m_blockProposerPtr->proposeMove();
    double dS = 0;
    double logLikelihoodRatio = m_sbmGraphPtr->getLogLikelihoodRatio(move);
    double logPriorRatio = m_sbmGraphPtr->getLogPriorRatio(move);
    double LogProposalProbRatio = m_blockProposerPtr->getLogProposalProbRatio(move);

    m_lastLogJointRatio = m_betaLikelihood * logLikelihoodRatio + m_betaPrior * logPriorRatio;
    m_lastLogAcceptance = LogProposalProbRatio + m_lastLogJointRatio;

    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        m_sbmGraphPtr->applyMove(move);
        m_blockProposerPtr->updateProbabilities(move);
    }
}

}
