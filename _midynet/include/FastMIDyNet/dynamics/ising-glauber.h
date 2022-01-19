#ifndef FAST_MIDYNET_ISING_MODEL_H
#define FAST_MIDYNET_ISING_MODEL_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class IsingGlauberDynamics: public BinaryDynamics {
    double m_couplingConstant;

    public:
        IsingGlauberDynamics(size_t numSteps, double couplingConstant, bool normalizeCoupling=true):
            BinaryDynamics(numSteps, normalizeCoupling), m_couplingConstant(couplingConstant) {}
        IsingGlauberDynamics(RandomGraph& randomGraph, size_t numSteps, double couplingConstant, bool normalizeCoupling=true):
            BinaryDynamics(randomGraph, numSteps, normalizeCoupling), m_couplingConstant(couplingConstant) {}

        const double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const override;
        const double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const override;
        const double getCoupling() const {
            if (m_normalizeCoupling)
                return m_couplingConstant / (2 * m_randomGraphPtr->getEdgeCount() / m_randomGraphPtr->getSize());
            else
                return m_couplingConstant;
        }
        void setCoupling(double couplingConstant) { m_couplingConstant = couplingConstant; }
};

} // namespace FastMIDyNet

#endif
