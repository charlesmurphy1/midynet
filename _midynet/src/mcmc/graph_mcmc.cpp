#include <iostream>
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

double RandomGraphMCMC::_getLogAcceptanceProb(const BlockMove& move) const{
    double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_randomGraphPtr->getLogLikelihoodRatioFromBlockMove(move);
    double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_randomGraphPtr->getLogPriorRatioFromBlockMove(move);
    m_lastLogJointRatio = logPriorRatio + logLikelihoodRatio;
    std::cout << logPriorRatio << " + " << logLikelihoodRatio << std::endl;
    std::cout << "HEre " << m_blockProposerPtr->getLogProposalProbRatio(move) + m_lastLogJointRatio << std::endl;
    return m_blockProposerPtr->getLogProposalProbRatio(move) + m_lastLogJointRatio;
}

bool RandomGraphMCMC::_doMetropolisHastingsStep() {
    std::cout << "Current partition: [";
    for (auto b : m_randomGraphPtr->getBlocks())
        std::cout << b << ", ";
    std::cout << "] with " << m_randomGraphPtr->getBlockCount() << std::endl;
    std::cout << "Proposing move" << std::endl;
    BlockMove move = m_blockProposerPtr->proposeMove();
    move.display() ;
    std::cout << "Computing acceptance prob: ";
    m_lastLogAcceptance = getLogAcceptanceProb(move);
    m_isLastAccepted = false;
    std::cout << "Accepting move: " << std::endl;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        applyBlockMove(move);
    }
    std::cout << (int)m_isLastAccepted << std::endl << std::endl;
    return m_isLastAccepted;
}

}
