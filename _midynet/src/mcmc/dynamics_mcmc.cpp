#include "FastMIDyNet/mcmc/dynamics_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

void DynamicsMCMC::setUp() {
    MCMC::setUp();
    m_randomGraphMCMC.setUp();
    m_edgeProposer.setUp(m_dynamics.getRandomGraph());
}

void DynamicsMCMC::doMetropolisHastingsStep() {
    if (m_uniform(rng) < m_sampleGraphPrior){
        m_lastMoveWasGraphMove = false;
        m_randomGraphMCMC.doMetropolisHastingsStep();
        m_lastLogJointRatio = m_randomGraphMCMC.getLastLogJointRatio();
        m_lastLogAcceptance = m_randomGraphMCMC.getLastLogAcceptance();
        m_isLastAccepted = m_randomGraphMCMC.isLastAccepted();
    }
    else {
        m_lastMoveWasGraphMove = true;
        GraphMove move = m_edgeProposer.proposeMove();
        double logLikelihoodRatio = m_dynamics.getLogLikelihoodRatio(move);
        double logPriorRatio = m_dynamics.getLogPriorRatio(move);
        double LogProposalProbRatio = m_edgeProposer.getLogProposalProbRatio(move);

        m_lastLogJointRatio = m_betaLikelihood * logLikelihoodRatio + m_betaPrior * logPriorRatio;
        m_lastLogAcceptance = LogProposalProbRatio + m_lastLogJointRatio;

        m_isLastAccepted = false;
        if (m_uniform(rng) < exp(m_lastLogAcceptance)){
            m_isLastAccepted = true;
            m_dynamics.applyMove(move);
            m_edgeProposer.updateProbabilities(move);
        }
    }

}

}
