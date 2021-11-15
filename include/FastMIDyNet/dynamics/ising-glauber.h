#ifndef FAST_MIDYNET_ISING_MODEL_H
#define FAST_MIDYNET_ISING_MODEL_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class IsingGlauberDynamic: public BinaryDynamics {
    double m_coupling_constant;

    public:
        IsingGlauberDynamic(RandomGraph& random_graph, RNG& rng, double coupling_constant):
            BinaryDynamics(random_graph, rng), m_coupling_constant(coupling_constant) {}

        double getActivationProb(const VertexNeighborhoodState& neighbor_state) const;
        double getDeactivationProb(const VertexNeighborhoodState& neighbor_state) const;
};

} // namespace FastMIDyNet

#endif
