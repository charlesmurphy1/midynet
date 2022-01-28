#include "FastMIDyNet/dynamics/degree.h"


namespace FastMIDyNet{

const double DegreeDynamics::computeActivationProb(const VertexNeighborhoodState& neighborhood_state) const{
    return (neighborhood_state[0] + neighborhood_state[1])/m_C;
}

const double DegreeDynamics::computeDeactivationProb(const VertexNeighborhoodState& neighborhood_state) const{
    return 1 - (neighborhood_state[0] + neighborhood_state[1])/m_C;
}

} // FastMIDyNet
