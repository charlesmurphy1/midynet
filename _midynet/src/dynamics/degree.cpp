#include "FastMIDyNet/dynamics/degree.h"


namespace FastMIDyNet{

const double DegreeDynamics::getActivationProb(const VertexNeighborhoodState& neighborhoodState) const{
    double p = (neighborhoodState[0] + neighborhoodState[1]) / m_C;
    return (1 - m_epsilon) * p + m_epsilon;
}

const double DegreeDynamics::getDeactivationProb(const VertexNeighborhoodState& neighborhoodState) const{
    double q = 1 - getActivationProb(neighborhoodState);
    return q;
}

} // FastMIDyNet
