#include "FastMIDyNet/mcmc/dynamics_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

double DynamicsMCMC::_getLogAcceptanceProb(const GraphMove& move) const {
    double logLikelihoodRatio = (m_betaLikelihood == 0) ? 0 : m_betaLikelihood * m_dynamicsPtr->getLogLikelihoodRatioFromGraphMove(move);
    double logPriorRatio = (m_betaPrior == 0) ? 0 : m_betaPrior * m_dynamicsPtr->getLogPriorRatioFromGraphMove(move);
    if (logLikelihoodRatio == -INFINITY or logPriorRatio == -INFINITY){
        m_lastLogJointRatio = -INFINITY;
        return -INFINITY;
    }
    m_lastLogJointRatio = logLikelihoodRatio + logPriorRatio;
    return m_randomGraphMCMCPtr->getLogProposalProbRatioFromGraphMove(move); + m_lastLogJointRatio;
}

bool DynamicsMCMC::_doMetropolisHastingsStep() {
    if (m_uniform(rng) < m_sampleGraphPriorProb){
        m_lastMoveWasGraphMove = false;
        m_randomGraphMCMCPtr->doMetropolisHastingsStep();
        m_lastLogJointRatio = m_randomGraphMCMCPtr->getLastLogJointRatio();
        m_lastLogAcceptance = m_randomGraphMCMCPtr->getLastLogAcceptance();
        return m_isLastAccepted = m_randomGraphMCMCPtr->isLastAccepted();
    }
    m_lastMoveWasGraphMove = true;
    GraphMove move = m_randomGraphMCMCPtr->proposeEdgeMove();
    m_lastLogAcceptance = getLogAcceptanceProb(move);
    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        applyGraphMove(move);
    }
    return m_isLastAccepted;

}

}
