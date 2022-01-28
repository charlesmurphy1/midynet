#ifndef FASTMIDYNET_DEGREE_DYNAMICS_H
#define FASTMIDYNET_DEGREE_DYNAMICS_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class DegreeDynamics: public BinaryDynamics {
    double m_C;

    public:
        DegreeDynamics(size_t numSteps, double C, bool normalizeCoupling=true, bool cache=false):
                BinaryDynamics(numSteps, normalizeCoupling, cache), m_C(C) {}
        DegreeDynamics(RandomGraph& random_graph, size_t numSteps, double C, bool normalizeCoupling=true, bool cache=false):
                BinaryDynamics(random_graph, numSteps, normalizeCoupling, cache), m_C(C) {}

        const double computeActivationProb(const VertexNeighborhoodState& neighborhood_state) const override;
        const double computeDeactivationProb(const VertexNeighborhoodState& neighborhood_state) const override;
        const double getC() const { return m_C; }
        void setC(double C) { m_C = C; }

    };

} // namespace FastMIDyNet

#endif
