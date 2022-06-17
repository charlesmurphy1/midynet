#ifndef FAST_MIDYNET_WILSON_COWAN_H
#define FAST_MIDYNET_WILSON_COWAN_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"
#include "FastMIDyNet/dynamics/glauber.h"


namespace FastMIDyNet{


class CowanDynamics: public BinaryDynamics {
private:
    double m_a;
    double m_nu;
    double m_mu;
    double m_eta;

public:
    CowanDynamics(
            size_t numSteps,
            double nu,
            double a=1,
            double mu=1,
            double eta=0.5,
            double autoActivationProb=1e-6,
            double autoDeactivationProb=0,
            bool normalizeCoupling=true,
            size_t numInitialActive=1):
        BinaryDynamics(
            numSteps,
            autoActivationProb,
            autoDeactivationProb,
            normalizeCoupling,
            numInitialActive),
        m_a(a),
        m_nu(nu),
        m_mu(mu),
        m_eta(eta) {}
    CowanDynamics(
            RandomGraph& randomGraph,
            size_t numSteps,
            double nu,
            double a=1,
            double mu=1,
            double eta=0.5,
            double autoActivationProb=1e-6,
            double autoDeactivationProb=0,
            bool normalizeCoupling=true,
            size_t numInitialActive=1):
        BinaryDynamics(
            randomGraph,
            numSteps,
            autoActivationProb,
            autoDeactivationProb,
            normalizeCoupling,
            numInitialActive),
        m_a(a),
        m_nu(nu),
        m_mu(mu),
        m_eta(eta) {}

    const double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const override {
        return sigmoid(m_a * ( getNu() * vertexNeighborState[1] - m_mu));
    }
    const double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const override{
        return m_eta;
    }
    const double getA() const { return m_a; }
    void setA(double a) { m_a = a; }
    const double getNu() const {
        if (m_normalizeCoupling)
            return m_nu / (2 * m_randomGraphPtr->getEdgeCount() / m_randomGraphPtr->getSize());
        else
            return m_nu;

    }
    void setNu(double nu) { m_nu = nu; }
    const double getMu() const { return m_mu; }
    void setMu(double mu) { m_mu = mu; }
    const double getEta() const { return m_eta; }
    void setEta(double eta) { m_eta = eta; }
};

} // namespace FastMIDyNet

#endif
