#ifndef FAST_MIDYNET_WILSON_COWAN_H
#define FAST_MIDYNET_WILSON_COWAN_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class CowanDynamics: public BinaryDynamics {
private:
    double m_a;
    double m_nu;
    double m_mu;
    double m_eta;

public:
    CowanDynamics(size_t numSteps, double nu, double a=1, double mu=1, double eta=0.5):
        BinaryDynamics(numSteps), m_a(a), m_nu(nu), m_mu(mu), m_eta(eta) {}
    CowanDynamics(RandomGraph& randomGraph, size_t numSteps, double nu, double a=1, double mu=1, double eta=0.5):
        BinaryDynamics(randomGraph, numSteps), m_a(a), m_nu(nu), m_mu(mu), m_eta(eta) {}

    double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const;
    double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const;
    double getA() const { return m_a; }
    void setA(double a) { m_a = a; }
    double getNu() const { return m_nu; }
    void setNu(double nu) { m_nu = nu; }
    double getMu() const { return m_mu; }
    void setMu(double mu) { m_mu = mu; }
    double getEta() const { return m_eta; }
    void setEta(double eta) { m_eta = eta; }
};

} // namespace FastMIDyNet

#endif
