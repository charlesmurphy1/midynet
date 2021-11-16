#include "FastMIDyNet/dynamics/ising-glauber.h"


static inline double sigmoid(double x) {
    return 1/(1+exp(-x));
}


namespace FastMIDyNet{

double IsingGlauberDynamics::getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const{
    return sigmoid(2*m_couplingConstant*(vertexNeighborState[0]-vertexNeighborState[1]));
}

double IsingGlauberDynamics::getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const{
    return sigmoid(-2*m_couplingConstant*(vertexNeighborState[0]-vertexNeighborState[1]));
}

} // FastMIDyNet
