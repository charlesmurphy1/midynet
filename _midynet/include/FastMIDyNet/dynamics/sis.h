#ifndef FAST_MIDYNET_SIS_DYNAMICS_H
#define FAST_MIDYNET_SIS_DYNAMICS_H


#include <vector>
#include <map>
#include <cmath>

#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class SISDynamics: public BinaryDynamics{

public:
        explicit SISDynamics(
                size_t numSteps,
                double infectionProb,
                double recoveryProb=0.5,
                double autoActivationProb=1e-6,
                double autoDeactivationProb=0,
                bool normalizeCoupling=true,
                size_t numInitialActive=1) :
            BinaryDynamics(
                numSteps,
                autoActivationProb,
                autoDeactivationProb,
                normalizeCoupling,
                numInitialActive),
            m_infectionProb(infectionProb),
            m_recoveryProb(recoveryProb){ }
        explicit SISDynamics(
                RandomGraph& randomGraph,
                size_t numSteps,
                double infectionProb,
                double recoveryProb=0.5,
                double autoActivationProb=1e-6,
                double autoDeactivationProb=0,
                bool normalizeCoupling=true,
                size_t numInitialActive=1) :
            BinaryDynamics(
                randomGraph,
                numSteps,
                autoActivationProb,
                autoDeactivationProb,
                normalizeCoupling,
                numInitialActive),
            m_infectionProb(infectionProb),
            m_recoveryProb(recoveryProb){ }

        const double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const override{
            return 1 - std::pow(1 - getInfectionProb(), vertexNeighborState[1]);
        }
        const double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const override{
            return m_recoveryProb;
        }

        const double getInfectionProb() const {
            if (not m_normalizeCoupling)
                return m_infectionProb;
            double infProb = m_infectionProb / (2 * m_randomGraphPtr->getEdgeCount() / m_randomGraphPtr->getSize());
            if (infProb > 1 - EPSILON)
                return 1 - EPSILON;
            if (infProb < 0)
                return 0;
            return infProb;
        }
        void setInfectionProb(double infectionProb) { m_infectionProb = infectionProb; }
        const double getRecoveryProb() const { return m_recoveryProb; }
        void setRecoveryProb(double recoveryProb) { m_recoveryProb = recoveryProb; }

private:
        double m_infectionProb, m_recoveryProb;
        const double EPSILON = 1e-6;
};

} // namespace FastMIDyNet

#endif
