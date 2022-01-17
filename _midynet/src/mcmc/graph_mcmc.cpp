#include <iostream>
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

void RandomGraphMCMC::doMetropolisHastingsStep() {
    m_blockProposerPtr->checkSafety();
    BlockMove move = m_blockProposerPtr->proposeMove();
    double dS = 0;
    double logLikelihoodRatio = m_randomGraphPtr->getLogLikelihoodRatioFromBlockMove(move);
    double logPriorRatio = m_randomGraphPtr->getLogPriorRatioFromBlockMove(move);
    double LogProposalProbRatio = m_blockProposerPtr->getLogProposalProbRatio(move);

    m_lastLogJointRatio = m_betaLikelihood * logLikelihoodRatio + m_betaPrior * logPriorRatio;
    m_lastLogAcceptance = LogProposalProbRatio + m_lastLogJointRatio;

    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        m_randomGraphPtr->applyBlockMove(move);
        m_blockProposerPtr->updateProbabilities(move);
    }
}

}
