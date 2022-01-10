#ifndef FAST_MIDYNET_ISING_MODEL_H
#define FAST_MIDYNET_ISING_MODEL_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class IsingGlauberDynamics: public BinaryDynamics {
    double m_couplingConstant;

    public:
        IsingGlauberDynamics(size_t numSteps, double couplingConstant):
            BinaryDynamics(numSteps), m_couplingConstant(couplingConstant) {}
        IsingGlauberDynamics(RandomGraph& randomGraph, size_t numSteps, double couplingConstant):
            BinaryDynamics(randomGraph, numSteps), m_couplingConstant(couplingConstant) {}

        double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const;
        double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const;
};

} // namespace FastMIDyNet

#endif
