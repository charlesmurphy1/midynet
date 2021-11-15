#include <cmath>

#include "FastMIDyNet/dynamics/SIS_dynamics.h"


namespace FastMIDyNet{

    double SISDynamic::getActivationProb(const VertexNeighborhoodState& neighbor_state) const {
        return (1 - m_autoInfectionProb)*(1 - std::pow(1 - m_infectionProb, neighbor_state[1])) + m_autoInfectionProb;
    }
    double SISDynamic::getDeactivationProb(const VertexNeighborhoodState& neighbor_state) const{
        return m_recoveryProb;
    }

} // namespace FastMIDyNet
