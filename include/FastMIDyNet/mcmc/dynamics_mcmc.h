#ifndef FAST_MIDYNET_DYNAMICS_MCMC_H
#define FAST_MIDYNET_DYNAMICS_MCMC_H

#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/proposer/edge_proposer/edge_proposer.h"

namespace FastMIDyNet{

class DynamicsMCMC{
private:
    Dynamics& m_dynamics;
    RandomGraphMCMC& m_randomGraphMCMC;
    EdgeProposer& m_edgeProposer;
public:
    DynamicsMCMC(Dynamics& dynamics, RandomGraphMCMC& randomGraphMCMC, EdgeProposer& edgeProposer):
    m_dynamics(dynamics), m_randomGraphMCMC(randomGraphMCMC), m_edgeProposer(edgeProposer) {}
};

}

#endif
