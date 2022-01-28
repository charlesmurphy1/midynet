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
                    bool normalizeCoupling=true,
                    bool cache=false) :
            BinaryDynamics(numSteps, normalizeCoupling, cache),
            m_infectionProb(infectionProb),
            m_recoveryProb(recoveryProb),
            m_autoInfectionProb(autoInfectionProb)  { }
        SISDynamics(RandomGraph& randomGraph,
                    size_t numSteps,
                    double infectionProb,
                    double recoveryProb=0.5,
                    double autoInfectionProb=1e-6,
                    bool normalizeCoupling=true,
                    bool cache=false) :
            BinaryDynamics(randomGraph, numSteps, normalizeCoupling, cache),
            m_infectionProb(infectionProb),
            m_recoveryProb(recoveryProb),
            m_autoInfectionProb(autoInfectionProb)  { }

        const double computeActivationProb(const VertexNeighborhoodState& vertexNeighborState) const override;
        const double computeDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const override;

        const double getInfectionProb() const {
            if (not m_normalizeCoupling)
                return m_infectionProb;
            double infProb = m_infectionProb / m_randomGraphPtr->getAverageDegree();
            if (infProb > 1 - 1E-5)
                return 1 - 1E-5;
            return infProb;
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
