#include "FastMIDyNet/mcmc/dynamics_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

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
        GraphMove move = m_randomGraphMCMCPtr->proposeEdgeMove();
        double logLikelihoodRatio = m_dynamicsPtr->getLogLikelihoodRatioFromGraphMove(move);
        double logPriorRatio = m_dynamicsPtr->getLogPriorRatioFromGraphMove(move);
        double LogProposalProbRatio = m_randomGraphMCMCPtr->getLogProposalProbRatio(move);

        m_lastLogJointRatio = m_betaLikelihood * logLikelihoodRatio + m_betaPrior * logPriorRatio;
        m_lastLogAcceptance = LogProposalProbRatio + m_lastLogJointRatio;

        m_isLastAccepted = false;
        if (m_uniform(rng) < exp(m_lastLogAcceptance)){
            m_isLastAccepted = true;
            m_dynamicsPtr->applyGraphMove(move);
            m_randomGraphMCMCPtr->updateProbabilities(move);
        }
    }

}

}
