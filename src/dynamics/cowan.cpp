#include "FastMIDyNet/dynamics/cowan.h"


static inline double sigmoid(double x) {
    return 1/(1+exp(-x));
}


namespace FastMIDyNet{

double CowanDynamics::getActivationProb(const VertexNeighborhoodState& neighborState) const{
    return sigmoid(m_a*(m_nu*neighborState[1] - m_mu));
}

double CowanDynamics::getDeactivationProb(const VertexNeighborhoodState& neighborState) const{
    return m_eta;
}

} // FastMIDyNet
