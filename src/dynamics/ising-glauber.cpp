#include "FastMIDyNet/dynamics/ising-glauber.h"


static inline double sigmoid(double x) {
    return 1/(1+exp(-x));
}


namespace FastMIDyNet{

double IsingGlauberDynamic::getActivationProb(const VertexNeighborhoodState& neighbor_state) const{
    return sigmoid(2*m_coupling_constant*(neighbor_state[0]-neighbor_state[1]));
}

double IsingGlauberDynamic::getDeactivationProb(const VertexNeighborhoodState& neighbor_state) const{
    return sigmoid(-2*m_coupling_constant*(neighbor_state[0]-neighbor_state[1]));
}

} // FastMIDyNet
