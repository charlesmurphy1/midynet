#ifndef FAST_MIDYNET_ISING_MODEL_H
#define FAST_MIDYNET_ISING_MODEL_H


#include "FastMIDyNet/data/dynamics/binary_dynamics.hpp"
#include "FastMIDyNet/data/util.h"


namespace FastMIDyNet{


template<typename GraphPriorType=RandomGraph>
class GlauberDynamics: public BinaryDynamics<GraphPriorType> {
    double m_couplingConstant;

    public:
        using BaseClass = BinaryDynamics<GraphPriorType>;

        GlauberDynamics(
                size_t numSteps,
                double couplingConstant,
                double autoActivationProb=0,
                double autoDeactivationProb=0,
                bool async=false,
                bool normalizeCoupling=true,
                int numInitialActive=-1):
            BaseClass(
                numSteps,
                autoActivationProb,
                autoDeactivationProb,
                async,
                normalizeCoupling,
                numInitialActive),
            m_couplingConstant(couplingConstant) {}
        GlauberDynamics(
                GraphPriorType& graphPrior,
                size_t numSteps,
                double couplingConstant,
                double autoActivationProb=0,
                double autoDeactivationProb=0,
                bool async=false,
                bool normalizeCoupling=true,
                int numInitialActive=-1):
            BaseClass(
                graphPrior,
                numSteps,
                autoActivationProb,
                autoDeactivationProb,
                async,
                normalizeCoupling,
                numInitialActive),
            m_couplingConstant(couplingConstant) {}

        const double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const override {
            double p = sigmoid( 2 * getCoupling() * ((int)vertexNeighborState[0] - (int)vertexNeighborState[1]));
            return p;
        }
        const double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const override {
            double p = sigmoid(-2 * getCoupling() * ((int)vertexNeighborState[0] - (int)vertexNeighborState[1]));
            return p;
        }
        const double getCoupling() const {
            if (not BaseClass::m_normalizeCoupling)
                return m_couplingConstant;
            double coupling = m_couplingConstant / (2 * BaseClass::m_graphPriorPtr->getEdgeCount() / BaseClass::m_graphPriorPtr->getSize());
            return coupling;
        }
        void setCoupling(double couplingConstant) { m_couplingConstant = couplingConstant; }
};

} // namespace FastMIDyNet

#endif
