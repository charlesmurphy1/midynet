#ifndef FAST_MIDYNET_ISING_MODEL_H
#define FAST_MIDYNET_ISING_MODEL_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class IsingGlauberDynamics: public BinaryDynamics {
    double m_couplingConstant;

    public:
        IsingGlauberDynamics(size_t numSteps, double couplingConstant, bool normalizeCoupling=true, bool cache=false):
            BinaryDynamics(numSteps, normalizeCoupling, cache), m_couplingConstant(couplingConstant) {}
        IsingGlauberDynamics(RandomGraph& randomGraph, size_t numSteps, double couplingConstant, bool normalizeCoupling=true, bool cache=false):
            BinaryDynamics(randomGraph, numSteps, normalizeCoupling, cache), m_couplingConstant(couplingConstant) {}

        const double computeActivationProb(const VertexNeighborhoodState& vertexNeighborState) const override;
        const double computeDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const override;
        const double getCoupling() const {
            if (not m_normalizeCoupling)
                return m_couplingConstant;
            double coupling = m_couplingConstant / m_randomGraphPtr->getAverageDegree();
            return coupling;
        }
        void setCoupling(double couplingConstant) { m_couplingConstant = couplingConstant; }
};

} // namespace FastMIDyNet

#endif
