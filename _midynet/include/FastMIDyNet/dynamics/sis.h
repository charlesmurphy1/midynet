#ifndef FAST_MIDYNET_SIS_DYNAMICS_H
#define FAST_MIDYNET_SIS_DYNAMICS_H


#include <vector>
#include <map>

#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class SISDynamics: public BinaryDynamics{

public:
        SISDynamics(size_t numSteps,
                    double infectionProb,
                    double recoveryProb=0.5,
                    double autoInfectionProb=1e-6,
                    bool normalizeCoupling=true) :
            BinaryDynamics(numSteps, normalizeCoupling),
            m_infectionProb(infectionProb),
            m_recoveryProb(recoveryProb),
            m_autoInfectionProb(autoInfectionProb)  { }
        SISDynamics(RandomGraph& randomGraph,
                    size_t numSteps,
                    double infectionProb,
                    double recoveryProb=0.5,
                    double autoInfectionProb=1e-6,
                    bool normalizeCoupling=true) :
            BinaryDynamics(randomGraph, numSteps, normalizeCoupling),
            m_infectionProb(infectionProb),
            m_recoveryProb(recoveryProb),
            m_autoInfectionProb(autoInfectionProb)  { }

        const double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const override;
        const double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const override;

        const double getInfectionProb() const {
            if (m_normalizeCoupling)
                return m_infectionProb / (2 * m_randomGraphPtr->getEdgeCount() / m_randomGraphPtr->getSize());
            else
                return m_infectionProb;
        }
        void setInfectionProb(double infectionProb) { m_infectionProb = infectionProb; }
        const double getRecoveryProb() const { return m_recoveryProb; }
        void setRecoveryProb(double recoveryProb) { m_recoveryProb = recoveryProb; }
        const double getAutoInfectionProb() const { return m_autoInfectionProb; }
        void setAutoInfectionProb(double autoInfectionProb) { m_autoInfectionProb = autoInfectionProb; }

private:
        double m_infectionProb, m_recoveryProb, m_autoInfectionProb;
};

} // namespace FastMIDyNet

#endif
