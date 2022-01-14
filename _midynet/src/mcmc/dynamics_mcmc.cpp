#include "FastMIDyNet/mcmc/dynamics_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

void DynamicsMCMC::setUp() {
    m_edgeProposerPtr->setUp(m_dynamicsPtr->getRandomGraph());
    m_randomGraphMCMCPtr->setRandomGraph(m_dynamicsPtr->getRandomGraphRef());
    m_randomGraphMCMCPtr->setUp();
    MCMC::setUp();
}

void DynamicsMCMC::doMetropolisHastingsStep() {
    if (m_uniform(rng) < m_sampleGraphPriorProb){
        m_lastMoveWasGraphMove = false;
        m_randomGraphMCMCPtr->doMetropolisHastingsStep();
        m_lastLogJointRatio = m_randomGraphMCMCPtr->getLastLogJointRatio();
        m_lastLogAcceptance = m_randomGraphMCMCPtr->getLastLogAcceptance();
        m_isLastAccepted = m_randomGraphMCMCPtr->isLastAccepted();
    }
    else {
        m_lastMoveWasGraphMove = true;
        GraphMove move = m_edgeProposerPtr->proposeMove();
        double logLikelihoodRatio = m_dynamicsPtr->getLogLikelihoodRatio(move);
        double logPriorRatio = m_dynamicsPtr->getLogPriorRatio(move);
        double LogProposalProbRatio = m_edgeProposerPtr->getLogProposalProbRatio(move);

        m_lastLogJointRatio = m_betaLikelihood * logLikelihoodRatio + m_betaPrior * logPriorRatio;
        m_lastLogAcceptance = LogProposalProbRatio + m_lastLogJointRatio;

        m_isLastAccepted = false;
        if (m_uniform(rng) < exp(m_lastLogAcceptance)){
            m_isLastAccepted = true;
            m_dynamicsPtr->applyMove(move);
            m_edgeProposerPtr->updateProbabilities(move);
        }
    }

}

}
