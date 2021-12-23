#include "FastMIDyNet/dynamics/degree.h"


namespace FastMIDyNet{

    double DegreeDynamics::getActivationProb(const VertexNeighborhoodState& neighborhood_state) const{
        return (neighborhood_state[0] + neighborhood_state[1])/m_C;
    }

    double DegreeDynamics::getDeactivationProb(const VertexNeighborhoodState& neighborhood_state) const{
        return 1 - (neighborhood_state[0] + neighborhood_state[1])/m_C;
    }

} // FastMIDyNet
