#ifndef FAST_MIDYNET_ISING_MODEL_H
#define FAST_MIDYNET_ISING_MODEL_H


#include "FastMIDyNet/dynamics/binary_dynamics.hpp"
#include "FastMIDyNet/dynamics/util.h"


namespace FastMIDyNet{


template<typename RandomGraphType=RandomGraph>
class GlauberDynamics: public BinaryDynamics<RandomGraphType> {
    double m_couplingConstant;

    public:
        using BaseClass = BinaryDynamics<RandomGraphType>;

        GlauberDynamics(
                size_t numSteps,
                double couplingConstant,
                double autoActivationProb=0,
                double autoDeactivationProb=0,
                bool normalizeCoupling=true,
                size_t numInitialActive=-1):
            BaseClass(
                numSteps,
                autoActivationProb,
                autoDeactivationProb,
                normalizeCoupling,
                numInitialActive),
            m_couplingConstant(couplingConstant) {}
        GlauberDynamics(
                RandomGraphType& randomGraph,
                size_t numSteps,
                double couplingConstant,
                double autoActivationProb=0,
                double autoDeactivationProb=0,
                bool normalizeCoupling=true,
                size_t numInitialActive=-1):
            BaseClass(
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
            if (not BaseClass::m_normalizeCoupling)
                return m_couplingConstant;
            double coupling = m_couplingConstant / (2 * BaseClass::m_randomGraphPtr->getEdgeCount() / BaseClass::m_randomGraphPtr->getSize());
            return coupling;
        }
        void setCoupling(double couplingConstant) { m_couplingConstant = couplingConstant; }
};

} // namespace FastMIDyNet

#endif
