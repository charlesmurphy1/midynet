#include "FastMIDyNet/dynamics/ising-glauber.h"


static inline double sigmoid(double x) {
    return 1/(1+exp(-x));
}


namespace FastMIDyNet{

double IsingGlauberDynamics::getActivationProb(const VertexNeighborhoodState& neighborState) const{
    return sigmoid(2*m_couplingConstant*(neighborState[0]-neighborState[1]));
}

double IsingGlauberDynamics::getDeactivationProb(const VertexNeighborhoodState& neighborState) const{
    return sigmoid(-2*m_couplingConstant*(neighborState[0]-neighborState[1]));
}

} // FastMIDyNet
