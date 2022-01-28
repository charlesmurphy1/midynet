#include <cmath>

#include "FastMIDyNet/dynamics/sis.h"


namespace FastMIDyNet{

    const double  SISDynamics::computeActivationProb(const VertexNeighborhoodState& vertexNeighborState) const {
        return (1 - m_autoInfectionProb)*(1 - std::pow(1 - getInfectionProb(), vertexNeighborState[1])) + m_autoInfectionProb;
    }
    const double  SISDynamics::computeDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const{
        return m_recoveryProb;
    }

} // namespace FastMIDyNet
