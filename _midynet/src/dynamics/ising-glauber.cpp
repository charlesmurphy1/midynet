#include "FastMIDyNet/dynamics/ising-glauber.h"


static inline double sigmoid(double x) {
    return 1/(1+exp(-x));
}


namespace FastMIDyNet{

const double IsingGlauberDynamics::getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const{
    return sigmoid( 2 * getCoupling() * (vertexNeighborState[0]-vertexNeighborState[1]));
}

const double IsingGlauberDynamics::getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const{
    return sigmoid(-2 * getCoupling() * (vertexNeighborState[0]-vertexNeighborState[1]));
}

} // FastMIDyNet
