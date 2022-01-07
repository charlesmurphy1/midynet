#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

void StochasticBlockGraphMCMC::setUp() {
    MCMC::setUp();
    m_blockProposer.setUp(m_sbmGraph);
}

void StochasticBlockGraphMCMC::doMetropolisHastingsStep() {
    BlockMove move = m_blockProposer.proposeMove();
    double dS = 0;
    double logLikelihoodRatio = m_sbmGraph.getLogLikelihoodRatio(move);
    double logPriorRatio = m_sbmGraph.getLogPriorRatio(move);
    double LogProposalProbRatio = m_blockProposer.getLogProposalProbRatio(move);

    m_lastLogJointRatio = m_betaLikelihood * logLikelihoodRatio + m_betaPrior * logPriorRatio;
    m_lastLogAcceptance = LogProposalProbRatio + m_lastLogJointRatio;

    m_isLastAccepted = false;
    if (m_uniform(rng) < exp(m_lastLogAcceptance)){
        m_isLastAccepted = true;
        m_sbmGraph.applyMove(move);
        m_blockProposer.updateProbabilities(move);
    }
}

}
