#include "FastMIDyNet/mcmc/dynamics_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"


namespace FastMIDyNet{

bool DynamicsMCMC::doMetropolisHastingsStep() {
    bool verbose=1;
    if (m_uniform(rng) < m_sampleGraphPriorProb){
        m_lastMoveWasGraphMove = false;
        m_randomGraphMCMC.doMetropolisHastingsStep();
        m_lastLogJointRatio = m_randomGraphMCMC.getLastLogJointRatio();
        m_lastLogAcceptance = m_randomGraphMCMC.getLastLogAcceptance();
        m_isLastAccepted = m_randomGraphMCMC.isLastAccepted();
    }
    else {
        m_lastMoveWasGraphMove = true;

        std::cout << "proposing move" << std::endl;

        GraphMove move = m_randomGraphMCMC.proposeEdgeMove();

        move.display();

        double logLikelihoodRatio = m_dynamics.getLogLikelihoodRatioFromGraphMove(move);
        if (m_betaLikelihood == 0)
            logLikelihoodRatio = 0;
        else
            logLikelihoodRatio *= m_betaLikelihood;

        std::cout << "logLikelihoodRatio = " << logLikelihoodRatio << std::endl;

        double logPriorRatio = m_dynamics.getLogPriorRatioFromGraphMove(move);
        if (m_betaPrior == 0)
            logPriorRatio = 0;
        else
            logPriorRatio *= m_betaPrior;

        std::cout << "logPriorRatio = " << logPriorRatio << std::endl;



        double LogProposalProbRatio = m_randomGraphMCMC.getLogProposalProbRatioFromGraphMove(move);


        std::cout << "LogProposalProbRatio = " << LogProposalProbRatio << std::endl;



        m_lastLogJointRatio = logLikelihoodRatio + logPriorRatio;

        if (logLikelihoodRatio == -INFINITY or logPriorRatio == -INFINITY)
            m_lastLogAcceptance = -INFINITY;
        else
            m_lastLogAcceptance = LogProposalProbRatio + m_lastLogJointRatio;
        m_isLastAccepted = false;


        std::cout << "Trying out move" << std::endl;

        if (m_uniform(rng) < exp(m_lastLogAcceptance)){

            std::cout << "move accepted" << std::endl;

            m_isLastAccepted = true;
            m_dynamics.applyGraphMove(move);
            m_randomGraphMCMC.updateProbabilitiesFromGraphMove(move);
        }

        std::cout << "finish" << std::endl;

        return m_isLastAccepted;
    }

}

}
