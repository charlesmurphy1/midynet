#ifndef FAST_MIDYNET_ISING_MODEL_H
#define FAST_MIDYNET_ISING_MODEL_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{

static inline double sigmoid(double x) {
    return 1/(1+exp(-x));
}


class GlauberDynamics: public BinaryDynamics {
    double m_couplingConstant;

    public:
        GlauberDynamics(
                size_t numSteps,
                double couplingConstant,
                double autoActivationProb=0,
                double autoDeactivationProb=0,
                bool normalizeCoupling=true,
                size_t numInitialActive=-1):
            BinaryDynamics(
                numSteps,
                autoActivationProb,
                autoDeactivationProb,
                normalizeCoupling,
                numInitialActive),
            m_couplingConstant(couplingConstant) {}
        GlauberDynamics(
                RandomGraph& randomGraph,
                size_t numSteps,
                double couplingConstant,
                double autoActivationProb=0,
                double autoDeactivationProb=0,
                bool normalizeCoupling=true,
                size_t numInitialActive=-1):
            BinaryDynamics(
                randomGraph,
                numSteps,
                autoActivationProb,
                autoDeactivationProb,
                normalizeCoupling,
                numInitialActive),
            m_couplingConstant(couplingConstant) {}

        const double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const override {
            return sigmoid( 2 * getCoupling() * (vertexNeighborState[0]-vertexNeighborState[1]));
        }
        const double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const override {
            return sigmoid(-2 * getCoupling() * (vertexNeighborState[0]-vertexNeighborState[1]));
        }
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
