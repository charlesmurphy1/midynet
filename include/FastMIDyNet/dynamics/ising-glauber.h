#ifndef FAST_MIDYNET_ISING_MODEL_H
#define FAST_MIDYNET_ISING_MODEL_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class IsingGlauberDynamics: public BinaryDynamics {
    double m_couplingConstant;

    public:
        IsingGlauberDynamics(RandomGraph& randomGraph, RNG& rng, double couplingConstant):
            BinaryDynamics(randomGraph, rng), m_couplingConstant(couplingConstant) {}

        double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const;
        double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const;
};

} // namespace FastMIDyNet

#endif
