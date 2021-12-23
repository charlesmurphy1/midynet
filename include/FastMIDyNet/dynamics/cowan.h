#ifndef FAST_MIDYNET_WILSON_COWAN_H
#define FAST_MIDYNET_WILSON_COWAN_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class CowanDynamics: public BinaryDynamics {
private:
    double m_a;
    double m_nu;
    double m_mu;
    double m_eta;

public:
    CowanDynamics(RandomGraph& randomGraph, size_t numSteps, double nu, double a=1, double mu=1, double eta=0.5):
        BinaryDynamics(randomGraph, numSteps), m_a(a), m_nu(nu), m_mu(mu), m_eta(eta) {}

    double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const;
    double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const;
};

} // namespace FastMIDyNet

#endif
