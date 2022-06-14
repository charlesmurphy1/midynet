#include <iostream>
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

double RandomGraphMCMC::_getLogAcceptanceProb(const BlockMove& move) const{
    double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_randomGraphPtr->getLogLikelihoodRatioFromBlockMove(move);
    double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_randomGraphPtr->getLogPriorRatioFromBlockMove(move);
    m_lastLogJointRatio = logPriorRatio + logLikelihoodRatio;
    return m_blockProposerPtr->getLogProposalProbRatio(move) + m_lastLogJointRatio;

}

bool RandomGraphMCMC::_doMetropolisHastingsStep() {
    BlockMove move = m_blockProposerPtr->proposeMove();
    m_lastLogAcceptance = getLogAcceptanceProb(move);
    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        applyBlockMove(move);
    }
    return m_isLastAccepted;
}

}
