#include <cmath>

#include "FastMIDyNet/dynamics/sis.h"


namespace FastMIDyNet{

    double SISDynamics::getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const {
        return (1 - m_autoInfectionProb)*(1 - std::pow(1 - m_infectionProb, vertexNeighborState[1])) + m_autoInfectionProb;
    }
    double SISDynamics::getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const{
        return m_recoveryProb;
    }

} // namespace FastMIDyNet
