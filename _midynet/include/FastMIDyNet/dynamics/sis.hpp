#ifndef FAST_MIDYNET_SIS_DYNAMICS_H
#define FAST_MIDYNET_SIS_DYNAMICS_H


#include <vector>
#include <map>
#include <cmath>

#include "FastMIDyNet/dynamics/binary_dynamics.hpp"


namespace FastMIDyNet{

template<typename RandomGraphType=RandomGraph>
class SISDynamics: public BinaryDynamics<RandomGraphType>{

public:
    using BaseClass = BinaryDynamics<RandomGraphType>;
    explicit SISDynamics(
            size_t numSteps,
            double infectionProb,
            double recoveryProb=0.5,
            double autoActivationProb=1e-6,
            double autoDeactivationProb=0,
            bool normalizeCoupling=true,
            size_t numInitialActive=1) :
        BaseClass(
            numSteps,
            autoActivationProb,
            autoDeactivationProb,
            normalizeCoupling,
            numInitialActive),
        m_infectionProb(infectionProb),
        m_recoveryProb(recoveryProb){ }
    explicit SISDynamics(
            RandomGraphType& randomGraph,
            size_t numSteps,
            double infectionProb,
            double recoveryProb=0.5,
            double autoActivationProb=1e-6,
            double autoDeactivationProb=0,
            bool normalizeCoupling=true,
            size_t numInitialActive=1) :
        BaseClass(
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
        if (not BaseClass::m_normalizeCoupling)
            return m_infectionProb;
        double infProb = m_infectionProb / (2 * BaseClass::m_graphPriorPtr->getEdgeCount() / BaseClass::m_graphPriorPtr->getSize());
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
