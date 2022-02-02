#ifndef FASTMIDYNET_DEGREE_DYNAMICS_H
#define FASTMIDYNET_DEGREE_DYNAMICS_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class DegreeDynamics: public BinaryDynamics {
    double m_C;
    double m_epsilon;

    public:
        DegreeDynamics(size_t numSteps, double C, double epsilon=1E-6):
                BinaryDynamics(numSteps, false), m_C(C), m_epsilon(epsilon) {}
        DegreeDynamics(RandomGraph& random_graph, size_t numSteps, double C, double epsilon=1e-6):
                BinaryDynamics(random_graph, numSteps, false), m_C(C), m_epsilon(epsilon) { }

        const double getActivationProb(const VertexNeighborhoodState& neighborhood_state) const override;
        const double getDeactivationProb(const VertexNeighborhoodState& neighborhood_state) const override;
        const double getC() const { return m_C; }
        const double getEpsilon() const { return m_epsilon; }
        void setC(double C) { m_C = C; }
        void setEpsilon(double epsilon) { m_epsilon = epsilon; }

    };

} // namespace FastMIDyNet

#endif
