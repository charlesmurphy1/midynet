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
        WilsonCowanDynamics(RandomGraph& random_graph, RNG& rng, double a, double nu, double mu, double eta):
            BinaryDynamics(random_graph, rng), m_a(a), m_nu(nu), m_mu(mu), m_eta(eta) {}

        double getActivationProb(const VertexNeighborhoodState& neighborhood_state) const;
        double getDeactivationProb(const VertexNeighborhoodState& neighborhood_state) const;
};

} // namespace FastMIDyNet

#endif
