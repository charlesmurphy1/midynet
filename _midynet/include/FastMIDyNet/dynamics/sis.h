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
                    double autoInfectionProb=1e-6) :
            BinaryDynamics(numSteps),
            m_infectionProb(infectionProb),
            m_recoveryProb(recoveryProb),
            m_autoInfectionProb(autoInfectionProb)  { }
        SISDynamics(RandomGraph& randomGraph,
                    size_t numSteps,
                    double infectionProb,
                    double recoveryProb=0.5,
                    double autoInfectionProb=1e-6) :
            BinaryDynamics(randomGraph, numSteps),
            m_infectionProb(infectionProb),
            m_recoveryProb(recoveryProb),
            m_autoInfectionProb(autoInfectionProb)  { }

        double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const;
        double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const;

        double getInfectionProb() const { return m_infectionProb; }
        void setInfectionProb(double infectionProb) { m_infectionProb = infectionProb; }
        double getRecoveryProb() const { return m_recoveryProb; }
        void setRecoveryProb(double recoveryProb) { m_recoveryProb = recoveryProb; }
        double getAutoInfectionProb() const { return m_autoInfectionProb; }
        void setAutoInfectionProb(double autoInfectionProb) { m_autoInfectionProb = autoInfectionProb; }

private:
        double m_infectionProb, m_recoveryProb, m_autoInfectionProb;
};

} // namespace FastMIDyNet

#endif
