#ifndef FAST_MIDYNET_WILSON_COWAN_H
#define FAST_MIDYNET_WILSON_COWAN_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class WilsonCowanDynamics: public BinaryDynamics {
    double m_a;
    double m_nu;
    double m_mu;
    double m_eta;

    public:
        WilsonCowanDynamics(RandomGraph& randomGraph, RNG& rng, double a, double nu, double mu, double eta):
            BinaryDynamics(randomGraph, rng), m_a(a), m_nu(nu), m_mu(mu), m_eta(eta) {}

        double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const;
        double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const;
};

} // namespace FastMIDyNet

#endif
