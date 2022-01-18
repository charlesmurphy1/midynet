#include <iostream>
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

void RandomGraphMCMC::doMetropolisHastingsStep() {
    BlockMove move = m_blockProposer.proposeMove();
    double dS = 0;
    double logLikelihoodRatio = m_randomGraphPtr->getLogLikelihoodRatioFromBlockMove(move);
    double logPriorRatio = m_randomGraphPtr->getLogPriorRatioFromBlockMove(move);
    double LogProposalProbRatio = m_blockProposer.getLogProposalProbRatio(move);

    m_lastLogJointRatio = m_betaLikelihood * logLikelihoodRatio + m_betaPrior * logPriorRatio;
    m_lastLogAcceptance = LogProposalProbRatio + m_lastLogJointRatio;

    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        m_randomGraphPtr->applyBlockMove(move);
        m_blockProposer.updateProbabilities(move);
    }
}

}
