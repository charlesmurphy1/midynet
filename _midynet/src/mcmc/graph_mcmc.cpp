#include <iostream>
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

void RandomGraphMCMC::doMetropolisHastingsStep() {
    BlockMove move = m_blockProposer.proposeMove();
    double dS = 0;
    double logLikelihoodRatio = m_randomGraphPtr->getLogLikelihoodRatioFromBlockMove(move);
    if (m_betaLikelihood == 0)
        logLikelihoodRatio = 0;
    else
        logLikelihoodRatio *= m_betaLikelihood;

    double logPriorRatio = m_randomGraphPtr->getLogPriorRatioFromBlockMove(move);
    if (m_betaPrior == 0)
        logPriorRatio = 0;
    else
        logPriorRatio *= m_betaPrior;

    double LogProposalProbRatio = m_blockProposer.getLogProposalProbRatio(move);

    m_lastLogJointRatio = logLikelihoodRatio + logPriorRatio;

    if (logLikelihoodRatio == -INFINITY or logPriorRatio == -INFINITY)
        m_lastLogAcceptance = -INFINITY;
    else
        m_lastLogAcceptance = LogProposalProbRatio + m_lastLogJointRatio;
    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        m_randomGraphPtr->applyBlockMove(move);
        m_blockProposer.updateProbabilities(move);
    }
}

}
