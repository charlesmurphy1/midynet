#include "FastMIDyNet/dynamics/wilson_cowan.h"


static inline double sigmoid(double x) {
    return 1/(1+exp(-x));
}


namespace FastMIDyNet{

double WilsonCowanDynamics::getActivationProb(const VertexNeighborhoodState& neighborState) const{
    return sigmoid(m_a*(m_nu*neighborState[1] - m_mu));
}

double WilsonCowanDynamics::getDeactivationProb(const VertexNeighborhoodState& neighborState) const{
    return m_eta;
}

} // FastMIDyNet
